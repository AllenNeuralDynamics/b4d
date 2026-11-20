"""Read part of a (planes, rows, cols) stack without loading the rest: .ims, .tif/.tiff, .npy, .h5/.hdf5, .zarr"""
import os
import numpy as np

GB = 1e9


class Volume:
    """shape and dtype, chunk (planes stored together, so reading fewer still decodes them all), decode_bytes (what the reader itself
    may hold while decoding) and read(z0, z1, y0, y1, x0, x1), or read(z0, z1) for whole planes. read_into(out, z0, z1), where a format
    has it, decodes whole planes into any array, a chunk at a time, without holding them all in memory first."""

    def __init__(self, path, shape, dtype, block, chunk=1, decode_bytes=0, read_into=None):
        self.path = str(path); self.shape = tuple(int(s) for s in shape); self.dtype = np.dtype(dtype); self.block = block
        self.chunk = int(chunk); self.decode_bytes = int(decode_bytes); self.read_into = read_into
        if len(self.shape) != 3:
            raise ValueError(f"{path}: expected (planes, rows, cols), got shape {self.shape}")

    def read(self, z0, z1, y0=0, y1=None, x0=0, x1=None):
        return self.block(z0, z1, y0, self.shape[1] if y1 is None else y1, x0, self.shape[2] if x1 is None else x1)


def _ims(path, channel, timepoint, max_ram):
    try:
        import imaris_io
    except ImportError:
        raise SystemExit("reading .ims needs the imaris-io package") from None
    v = imaris_io.ImsVolume(path, channel=channel, timepoint=timepoint, resolution_level=0, threads=1)
    # each decoding thread keeps up to two chunks, which adds up at this chunk size: give them a quarter of the limit
    each = 2 * int(np.prod(v.chunks)) * v.dtype.itemsize
    threads = max(1, min(os.cpu_count() or 1, 64, int(.25 * max_ram * GB // each) if max_ram else 64))
    if threads > 1:
        v.close(); v = imaris_io.ImsVolume(path, channel=channel, timepoint=timepoint, resolution_level=0, threads=threads)
    return Volume(path, v.shape, v.dtype, lambda z0, z1, y0, y1, x0, x1: v[z0:z1, y0:y1, x0:x1], v.chunks[0], threads * each,
                  lambda out, z0, z1: v.read_into(out, z0, z1, 0, v.shape[1], 0, v.shape[2]))


def _tiff(path):
    import tifffile
    t = tifffile.TiffFile(path)
    series = t.series[0] if len(t.series) else None
    pages = series.pages if series is not None else t.pages
    first = pages[0]; shape = (len(pages), *first.shape)
    if len(shape) != 3:                                                 # e.g. colour pages, or one page holding the stack
        squeezed = tuple(s for s in (series.shape if series is not None else shape) if s > 1)
        if len(squeezed) != 3:
            raise ValueError(f"{path}: expected (planes, rows, cols), got shape {shape}")
        a = np.squeeze(t.asarray())
        return Volume(path, a.shape, a.dtype, lambda z0, z1, y0, y1, x0, x1: a[z0:z1, y0:y1, x0:x1])
    try:
        m = tifffile.memmap(path, mode="r")                             # uncompressed and contiguous: only what is touched gets read
        if m.shape != shape:
            raise ValueError
        return Volume(path, shape, m.dtype, lambda z0, z1, y0, y1, x0, x1: np.array(m[z0:z1, y0:y1, x0:x1]))
    except (ValueError, OSError):
        pass
    return Volume(path, shape, first.dtype, lambda z0, z1, y0, y1, x0, x1: np.stack([pages[k].asarray()[y0:y1, x0:x1] for k in range(z0, z1)]))


def open_volume(path, h5_dataset="data", channel=0, timepoint=0, max_ram=None):
    low = str(path).lower().rstrip("/\\")
    if low.endswith(".ims"):
        return _ims(path, channel, timepoint, max_ram)
    if low.endswith((".tif", ".tiff")):
        return _tiff(path)
    if low.endswith(".npy"):
        a = np.load(path, mmap_mode="r"); chunk = 1
    elif low.endswith((".h5", ".hdf5")):
        import h5py
        a = h5py.File(path, "r")[h5_dataset]; chunk = a.chunks[0] if a.chunks else 1
    elif low.endswith(".zarr"):
        import zarr
        a = zarr.open(path, mode="r"); chunk = a.chunks[0]
    else:
        raise ValueError(f"unknown stack format: {path}")
    return Volume(path, a.shape, a.dtype, lambda z0, z1, y0, y1, x0, x1: np.asarray(a[z0:z1, y0:y1, x0:x1]), chunk)
