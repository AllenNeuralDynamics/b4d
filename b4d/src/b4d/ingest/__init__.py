"""b4d.ingest - the zero-copy block interface between frame sources and the B4D writer.

    from b4d.ingest import BlockSource, FrameBlock, SourceInfo, SourceStats, validate_source   # the contract
    from b4d.ingest import SimulatedCameraSource                                             # a source without hardware
    from b4d.ingest import BlockRecorder, RecorderConfig                                     # the writer side

A camera vendor package implements BlockSource (structurally, no import of b4d needed) and hands blocks of frames to
BlockRecorder, which compresses them in place with the native CPU writer or the CUDA encoder. See protocol.py for the
contract, the egrabber repository for a real implementation, and examples/ in this repository for minimal ones.
"""

from .protocol import FRAME_DTYPE, BlockSource, FrameBlock, SourceInfo, SourceStats, validate_source
from .recorder import CSV_FIELDS, BlockRecorder, RecorderConfig, append_csv, load_source
from .simulated import SimulatedCameraSource, synthetic_frames

__all__ = [
    "FRAME_DTYPE",
    "BlockSource",
    "FrameBlock",
    "SourceInfo",
    "SourceStats",
    "validate_source",
    "SimulatedCameraSource",
    "synthetic_frames",
    "BlockRecorder",
    "RecorderConfig",
    "append_csv",
    "CSV_FIELDS",
    "load_source",
]
