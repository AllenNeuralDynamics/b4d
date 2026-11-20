"""The zero-copy block interface between a frame source (a camera driver, or a simulator) and the B4D writer.

A *source* owns a small pool of large, contiguous block buffers, each holding ``frames_per_block`` frames. The
producer (a frame grabber's DMA engine, or a simulator) fills the blocks in order; the consumer takes a completed
block, compresses straight out of its memory and hands the block back. Nothing is copied between the producer
and the compressor, which is why the writer can keep up with multi-GB/s cameras.

b4d defines the contract here and consumes it in :class:`b4d.ingest.BlockRecorder`. A camera-vendor package
implements it (see the ``egrabber`` repository for the Euresys Coaxlink implementation) **without importing b4d**:
the contract is structural (:class:`typing.Protocol`), so any object with these attributes and methods works.
:func:`validate_source` checks an object against the contract at run time.

Contract
--------
Construction (by convention, so that generic tools can instantiate a source from a name and keyword arguments)::

    Source(frames_per_block=64, nblocks=3, **vendor_specific_kwargs)
    # optional: buffers=[...]  caller-provided block memory (numpy or CuPy arrays of >= frames_per_block frames each),
    #           for a higher layer that wants the frames to land in its own buffers (GPU memory, a writer's slots).

Memory ownership: the *vendor package owns the pool*. It allocates, pins/announces, revokes
and frees the block memory, because only it knows when its SDK has stopped writing (freeing or reusing memory the SDK
may still DMA into is a use-after-free) and which allocation each SDK accepts. b4d never allocates frame memory for a
source and requires no vendor to accept external memory. The optional ``buffers=`` argument above is a convenience for
vendors that can take caller memory, not part of the required contract. A vendor whose SDK cannot deliver contiguous
zero-copy blocks copies into its own pool inside its package and reports that in ``SourceStats``. The consumer may
page-lock a vendor block it was given (``block_array``), for instance with ``cudaHostRegister``, but does not free it.

Attributes and methods::

    source.info                 -> SourceInfo         (available right after construction; may open the device briefly)
    source.frames_per_block     -> int
    source.nblocks              -> int
    source.start(max_frames)    -> None               allocate/pin the pool if not yet done and start producing;
                                                      the source stops by itself after max_frames (None = until stop())
    source.wait_block(timeout_ms) -> FrameBlock | None   None on timeout; raises StopIteration once stopped and drained
    source.release(block)       -> None               the block's memory may be refilled from now on
    source.stop()               -> None               stop producing; a final partial block may still be delivered
    source.stats()              -> SourceStats
    source.close()              -> None               free everything (also a context manager)
    source.block_array(i)       -> array              optional: the full (frames_per_block, H, W) array of pool block i
                                                      (lets the consumer page-lock the memory for a GPU, for instance)

A :class:`FrameBlock` is valid from ``wait_block`` until ``release``: the consumer must not keep references to
``block.array`` after releasing. Blocks are delivered in acquisition order; ``seq`` counts completed blocks from 0.

Loss accounting is the source's job: ``SourceStats.lost`` counts frames the producer generated that never reached a
block (for instance because every block was held by the consumer), ``incomplete`` frames that arrived damaged. A
recording is *sustained* when both stay zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

#: per-frame metadata table carried by every block: at least these fields, in this order
FRAME_DTYPE = np.dtype(
    [
        ("frame_id", "<u8"),  # producer's frame counter
        ("timestamp_ns", "<u8"),  # producer's clock (camera timestamp, or the simulator's pacing clock)
        ("host_ns", "<u8"),  # host monotonic clock when the frame became available
        ("size_filled", "<u8"),  # bytes delivered into the slot
        ("incomplete", "<u4"),  # 1 if the producer flagged the frame incomplete
        ("slot", "<u4"),  # position inside the block
    ]
)


@dataclass
class SourceInfo:
    """What the writer needs to know about the frames."""

    name: str  # vendor + model, free text
    serial: str
    shape: Tuple[int, int]  # (height, width) of one frame
    dtype: Any  # numpy dtype of a pixel (uint16 or uint8)
    fps: float  # nominal frames per second (the writer sizes buffers and run lengths from it)
    frame_bytes: int  # bytes per frame as delivered (container size, e.g. 2 bytes/px for 12-in-16 bit)
    camera_model: Optional[Tuple[float, float, float]] = None  # (conversion DN/e-, offset DN, read noise e-), if known
    extra: Dict[str, Any] = field(default_factory=dict)  # vendor fields (pixel format, exposure, link, ...)

    def asdict(self) -> dict:
        d = dict(self.__dict__)
        d["dtype"] = str(np.dtype(self.dtype))
        return d


@dataclass
class SourceStats:
    delivered: int = 0  # frames delivered into blocks since start()
    lost: int = 0  # frames the producer generated that never reached a block
    incomplete: int = 0  # frames delivered damaged
    blocks_completed: int = 0
    blocks_released: int = 0
    free_blocks: int = 0  # blocks available to the producer right now
    ready_blocks: int = 0  # completed blocks waiting for wait_block()
    fps_measured: float = 0.0  # from the producer's timestamps
    extra: Dict[str, Any] = field(default_factory=dict)


class FrameBlock:
    """A completed block. Sources may return this class or any object with the same attributes.

    array      : (nframes, H, W) view over the block's memory (numpy for host memory, CuPy for device memory)
    full_array : the whole (frames_per_block, H, W) buffer (array is its first nframes)
    frames     : structured array (FRAME_DTYPE) with one row per frame
    seq        : 0-based sequence number of the block in the acquisition
    index      : pool slot of the block (passed back in release)
    nframes    : frames in the block (== frames_per_block except for a final partial block)
    partial    : True for a final partial block after a stop
    """

    __slots__ = ("array", "full_array", "frames", "seq", "index", "nframes", "partial")

    def __init__(
        self, array, frames, seq: int, index: int, nframes: Optional[int] = None, partial: bool = False, full_array=None
    ):
        self.full_array = array if full_array is None else full_array
        self.nframes = int(nframes if nframes is not None else array.shape[0])
        self.array = self.full_array[: self.nframes]
        self.frames = frames
        self.seq = int(seq)
        self.index = int(index)
        self.partial = bool(partial)

    @property
    def timestamps_ns(self):
        return self.frames["timestamp_ns"]

    def __repr__(self):
        return f"FrameBlock(seq={self.seq}, pool={self.index}, n={self.nframes}{', partial' if self.partial else ''})"


@runtime_checkable
class BlockSource(Protocol):
    """Structural type of a frame source (see the module docstring for the full contract)."""

    info: SourceInfo
    frames_per_block: int
    nblocks: int

    def start(self, max_frames: Optional[int] = None) -> None: ...

    def wait_block(self, timeout_ms: int = 1000): ...

    def release(self, block) -> None: ...

    def stop(self) -> None: ...

    def stats(self) -> SourceStats: ...

    def close(self) -> None: ...


_REQUIRED_METHODS = ("start", "wait_block", "release", "stop", "stats", "close")
_REQUIRED_INFO = ("name", "serial", "shape", "dtype", "fps", "frame_bytes")


def validate_source(source) -> SourceInfo:
    """Check an object against the BlockSource contract; returns its SourceInfo. Raises TypeError with the first
    missing piece, so a new vendor implementation gets a precise message instead of a failure deep in the writer."""
    for m in _REQUIRED_METHODS:
        if not callable(getattr(source, m, None)):
            raise TypeError(f"block source {type(source).__name__} lacks method {m}()")
    for a in ("frames_per_block", "nblocks"):
        if not isinstance(getattr(source, a, None), int) or getattr(source, a) < 1:
            raise TypeError(f"block source {type(source).__name__} needs a positive int attribute {a}")
    info = getattr(source, "info", None)
    if info is None:
        raise TypeError(f"block source {type(source).__name__} has no .info")
    _validate_info(info)
    return info


def _validate_info(info) -> None:
    for a in _REQUIRED_INFO:
        if not hasattr(info, a):
            raise TypeError(f"source.info lacks {a}")
    if len(tuple(info.shape)) != 2:
        raise TypeError("source.info.shape must be (height, width)")
    if np.dtype(info.dtype) not in (np.dtype("uint16"), np.dtype("uint8")):
        raise TypeError("source.info.dtype must be uint16 or uint8")
    if not (info.fps > 0):
        raise TypeError("source.info.fps must be positive")
