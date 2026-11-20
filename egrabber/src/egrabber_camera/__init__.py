"""egrabber_camera - Euresys eGrabber (Coaxlink) camera driver with a zero-copy block interface.

    from egrabber_camera import EGrabberSource            # the BlockSource b4d (or anything else) consumes
    from egrabber_camera import Ingest, Block, Stats       # the lower-level ctypes engine binding
    from egrabber_camera import IngestDevice               # SDK-shaped feature access + device-only use

The package has no dependency on b4d: it only produces blocks of frames in pinned memory. b4d's
BlockRecorder consumes them. See README.md for the contract and the examples.
"""

from .genicam import IngestDevice, Remote, Stream
from .ingest import FRAME_DTYPE, Block, CameraInfo, Ingest, IngestError, OrderError, Stats, find_dll
from .source import CAMERA_MODELS, EGrabberSource, SourceInfo, SourceStats

__all__ = [
    "EGrabberSource",
    "SourceInfo",
    "SourceStats",
    "CAMERA_MODELS",
    "Ingest",
    "IngestDevice",
    "Remote",
    "Stream",
    "Block",
    "Stats",
    "CameraInfo",
    "IngestError",
    "OrderError",
    "FRAME_DTYPE",
    "find_dll",
]
__version__ = "0.1.0"
