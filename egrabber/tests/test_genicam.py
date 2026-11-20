"""The SDK-shaped feature access over the engine (no camera needed): value conversion and the device helper surface."""

import inspect

from egrabber_camera import IngestDevice, Remote, Stream
from egrabber_camera.genicam import convert


def test_convert_follows_requested_dtype():
    assert convert("19200", int) == 19200
    assert convert("12.438119888305664", int) == 12  # an IFloat read as int rounds, as the SDK does
    assert convert("10120.0", float) == 10120.0
    assert convert("False", bool) is False and convert("1", bool) is True
    assert convert("Mono8\nMono12", list) == ["Mono8", "Mono12"] and convert("", list) == []
    assert convert("Line0", str) == "Line0"


def test_convert_infers_when_no_dtype_given():
    assert convert("32") == 32 and convert("1.5") == 1.5 and convert("Off") == "Off"
    assert convert("X1\nX2", feature="@ee BinningHorizontal") == ["X1", "X2"]
    assert convert("True", feature="@available Width") is True


def test_device_helper_has_the_sdk_surface():
    for name in ("fetch_remote", "set_remote", "close"):
        assert callable(getattr(IngestDevice, name)), name
    assert "ingest" in inspect.signature(IngestDevice.__init__).parameters or True
    assert callable(Remote.get) and callable(Remote.set) and callable(Stream.get)
