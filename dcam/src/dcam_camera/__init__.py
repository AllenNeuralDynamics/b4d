"""dcam_camera - Hamamatsu DCAM-API camera driver with a zero-copy block interface.

    from dcam_camera import DcamSource                  # the BlockSource b4d (or anything else) consumes
    from dcam_camera import Dcam, open_by_serial        # the DCAM-API adapter (property access by name)

The package has no dependency on b4d: it produces blocks of frames in host memory that DCAM writes into
directly (dcambuf_attach). b4d's BlockRecorder consumes them. See README.md for the contract and the examples.
"""

from .dcam import Dcam, DCAMError, Prop, enumerate_devices, load, open_by_serial
from .source import (
    CAMERA_MODELS,
    FRAME_DTYPE,
    GPU_STATUS,
    DcamBlock,
    DcamSource,
    SourceInfo,
    SourceStats,
    aligned_empty,
    unpack_mono12,
)

__all__ = [
    "DcamSource",
    "DcamBlock",
    "SourceInfo",
    "SourceStats",
    "FRAME_DTYPE",
    "CAMERA_MODELS",
    "GPU_STATUS",
    "Dcam",
    "DCAMError",
    "Prop",
    "enumerate_devices",
    "load",
    "open_by_serial",
    "aligned_empty",
    "unpack_mono12",
]
__version__ = "0.1.0"
