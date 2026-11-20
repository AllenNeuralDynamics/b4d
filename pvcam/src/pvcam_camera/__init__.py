"""pvcam_camera - Teledyne Photometrics (PVCAM) camera driver with a zero-copy block interface.

    from pvcam_camera import PvcamSource          # the BlockSource b4d (or anything else) consumes
    from pvcam_camera import Pvcam, PvcamError    # the ctypes binding to the PVCAM runtime

PVCAM writes frames into one circular buffer this package allocates; blocks are views of that buffer. The package
does not depend on b4d. See README.md for the contract, the buffer handling and the examples.
"""

from .pvcam import FrameRecord, Pvcam, PvcamError, Region, find_library
from .source import (
    CAMERA_MODELS,
    FRAME_DTYPE,
    FRAME_INFO_DTYPE,
    GPU_STATUS,
    Block,
    PvcamSource,
    SourceInfo,
    SourceStats,
)

__all__ = [
    "PvcamSource",
    "SourceInfo",
    "SourceStats",
    "Block",
    "CAMERA_MODELS",
    "FRAME_DTYPE",
    "FRAME_INFO_DTYPE",
    "GPU_STATUS",
    "Pvcam",
    "PvcamError",
    "FrameRecord",
    "Region",
    "find_library",
]
__version__ = "0.1.0"
