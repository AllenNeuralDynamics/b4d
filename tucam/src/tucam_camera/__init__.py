"""tucam_camera - Tucsen camera driver (TUCam SDK V3) with a zero-copy block interface.

    from tucam_camera import TucamSource          # the BlockSource b4d (or anything else) consumes
    from tucam_camera import TUCam, devices, load  # the thin adapter over the SDK's own Python binding

The package has no dependency on b4d: it only produces blocks of frames in host memory the camera writes into.
b4d's BlockRecorder consumes them. See README.md for the contract, the zero-copy layout and the examples.
"""

from .source import FRAME_DTYPE, GPU_UNAVAILABLE, FrameBlock, SourceInfo, SourceStats, TucamSource
from .tucam import (
    PFNC_MONO8,
    PFNC_MONO12,
    PFNC_MONO12_PACKED,
    PFNC_MONO16,
    PIXEL_FORMATS,
    TUCam,
    TUCamError,
    devices,
    enumerate_devices,
    install,
    load,
    resolve_root,
)

__all__ = [
    "TucamSource",
    "SourceInfo",
    "SourceStats",
    "FrameBlock",
    "FRAME_DTYPE",
    "GPU_UNAVAILABLE",
    "TUCam",
    "TUCamError",
    "devices",
    "enumerate_devices",
    "install",
    "load",
    "resolve_root",
    "PIXEL_FORMATS",
    "PFNC_MONO8",
    "PFNC_MONO12",
    "PFNC_MONO16",
    "PFNC_MONO12_PACKED",
]
__version__ = "0.1.0"
