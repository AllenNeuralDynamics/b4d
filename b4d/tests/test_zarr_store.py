"""Native zarr v3 sharded writer (b4d.zarr_store + b4d_zarr.c)."""
import numpy as np
import pytest

import b4d
import b4d.zarr_store as zs
from b4d.zarr3 import B4D

zarr = pytest.importorskip("zarr", minversion="3")
numcodecs = pytest.importorskip("numcodecs")
CAM = b4d.Camera(2.0, 100.0, 3.0, exact_vst=False)   # writer tests: bounds in Anscombe sigma (test_exact_vst covers the curve)
RNG = np.random.default_rng(1)


def synth(shape):
    e = RNG.poisson(5 + 40 * (RNG.random(shape) > 0.9)) + RNG.normal(0, 3, shape)
    return np.clip(np.round(100 + 2 * e), 0, 65535).astype(np.uint16)


def roundtrip(tmp_path, img, codec, shards, chunks, threads=3, writers=1, blocks=None):
    path = str(tmp_path / "a.zarr")
    arr = zs.create(path, img.shape, dtype=img.dtype, shards=shards, chunks=chunks, codec=codec, threads=threads, writers=writers)
    z = img.shape[0]; depth = shards[0]
    futs = [arr[z0:min(z0 + depth, z)].write(img[z0:min(z0 + depth, z)]) for z0 in range(0, z, depth)]
    total = sum(f.result() for f in futs); arr.close()
    back = zarr.open_array(path, mode="r")[:]
    return back, total, path


@pytest.mark.parametrize("shape,shards,chunks", [((32, 256, 256), (16, 128, 128), (16, 64, 64)),
                                                   ((40, 300, 700), (16, 128, 256), (16, 64, 64)),       # partial everything
                                                   ((7, 65, 130), (8, 64, 64), (8, 32, 32))])              # thinner than one shard
def test_b4d_lossy_and_lossless(tmp_path, shape, shards, chunks):
    img = synth(shape)
    back, total, _ = roundtrip(tmp_path / "l", img, B4D(q=1, mode=2, camera=CAM, predictor="above", entropy="rice"), shards, chunks)
    err = np.abs(back.astype(float) - img) / CAM.sigma(img)
    assert err.max() <= 0.5 + 1e-6 and back.shape == img.shape
    back, total, _ = roundtrip(tmp_path / "ll", img, B4D(q=0, camera=CAM, predictor="left_above", entropy="ans"), shards, chunks)
    assert np.array_equal(back, img)


def test_identical_to_python_codec(tmp_path):
    """chunks written natively must equal what zarr-python's B4D codec would write for the same chunk"""
    img = synth((16, 128, 128))
    codec = B4D(q=1, mode=2, camera=CAM, predictor="above", entropy="rice")
    _, _, path = roundtrip(tmp_path, img, codec, (16, 128, 128), (16, 64, 64))
    import struct
    shard = open(f"{path}/c/0/0/0", "rb").read()
    n = 4; idx = shard[-(n * 16 + 4):-4]
    offs = struct.unpack("<" + "QQ" * n, idx)
    first = shard[offs[0]: offs[0] + offs[1]]
    ref = b4d.compress(img[:16, :64, :64], q=1, mode=2, camera=CAM, predictor="above", entropy="rice")
    assert first == ref


@pytest.mark.parametrize("codec", ["raw", numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE),
                                   numcodecs.Zstd(level=3), numcodecs.GZip(level=3)])
def test_other_codecs_lossless(tmp_path, codec):
    img = synth((24, 130, 200))
    back, total, path = roundtrip(tmp_path, img, codec, (8, 64, 128), (8, 32, 64), threads=2)
    assert np.array_equal(back, img)


def _blosc_patterns():
    rng = np.random.default_rng(7)
    shape = (24, 130, 200)
    noisy = synth(shape)
    dark = rng.poisson(0.4, shape).astype(np.uint16)             # high bytes mostly zero: Blosc2 run-length streams
    mixed = noisy.copy(); mixed[:, :64] = 3                        # constant and noisy streams in one tile
    return {"noisy": noisy, "dark": dark, "const": np.full(shape, 7, np.uint16), "zeros": np.zeros(shape, np.uint16), "mixed": mixed}


@pytest.mark.parametrize("cname,shuffle,clevel", [("lz4", numcodecs.Blosc.SHUFFLE, 5), ("lz4", numcodecs.Blosc.BITSHUFFLE, 9),
                                                  ("lz4hc", numcodecs.Blosc.SHUFFLE, 5), ("zstd", numcodecs.Blosc.SHUFFLE, 5),
                                                  ("zstd", numcodecs.Blosc.BITSHUFFLE, 1), ("zstd", numcodecs.Blosc.NOSHUFFLE, 7)])
def test_blosc_native_on_pool_is_blosc1_readable(tmp_path, cname, shuffle, clevel):
    """zarr 'blosc' chunks (lz4/zstd) are compressed on the C pool from per-worker Blosc2 contexts and rewritten in the
    Blosc1 layout: numcodecs (c-blosc1) must decode every stream kind, including Blosc2's run-length and all-zero cases,
    and none of them may have needed the uncompressed fallback"""
    codec = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=shuffle)
    for name, img in _blosc_patterns().items():
        path = str(tmp_path / f"{name}.zarr")
        arr = zs.create(path, img.shape, dtype=img.dtype, shards=(8, 64, 128), chunks=(8, 32, 64), codec=codec, threads=3, writers=1)
        assert arr._kind == 2 and arr._blosc[0] == cname                     # the native path, not Python-thread numcodecs
        total = sum(arr[z0:z0 + 8].write(img[z0:z0 + 8]).result() for z0 in range(0, 24, 8)); arr.close()
        assert arr.stats.get("blosc1_fallback", 0) == 0, name
        back = zarr.open_array(path, mode="r")
        assert [c["name"] for c in back.metadata.to_dict()["codecs"][0]["configuration"]["codecs"]] == ["bytes", "blosc"]
        assert np.array_equal(back[:], img), name
        if name in ("const", "zeros"):
            assert total < img.nbytes / 50, (name, total)                     # runs really were encoded, not stored raw


def test_blosc_native_matches_numcodecs_decode_of_single_chunk(tmp_path):
    """each native chunk is a self-contained Blosc1 buffer numcodecs decodes to the same bytes"""
    import struct
    img = synth((8, 64, 128))
    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    _, _, path = roundtrip(tmp_path, img, codec, (8, 64, 128), (8, 32, 64), threads=2)
    shard = open(f"{path}/c/0/0/0", "rb").read()
    n = 4; offs = struct.unpack("<" + "QQ" * n, shard[-(n * 16 + 4):-4])
    first = shard[offs[0]: offs[0] + offs[1]]
    assert first[0] == 2 and first[2] & 0x2 == 0                            # Blosc1 format version, not memcpyed
    assert np.frombuffer(codec.decode(first), np.uint16).tobytes() == img[:8, :32, :64].tobytes()


def test_windows_reopen_alignment(tmp_path):
    img = synth((32, 256, 300))
    path = str(tmp_path / "w.zarr")
    spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": path},
            "metadata": {"shape": list(img.shape), "data_type": "uint16", "chunk_grid": {"configuration": {"chunk_shape": [16, 128, 128]}},
                         "codecs": [{"name": "sharding_indexed", "configuration": {"chunk_shape": [16, 64, 64],
                                     "codecs": [{"name": "b4d", "configuration": {"q": 1, "mode": 2, "conversion": 2.0, "offset": 100.0,
                                                                                   "read_noise": 3.0, "predictor": "above", "entropy": "rice"}}]}}]}}
    arr = zs.open(spec, create=True, threads=2, writers=2)
    arr[0:16, 0:128].write(img[0:16, 0:128]); arr[0:16, 128:256].write(img[0:16, 128:256]); arr[16:32].write(img[16:32])
    arr.close()
    arr2 = zs.open(path)
    assert arr2.shape == img.shape and arr2.shards == (16, 128, 128) and arr2.chunks == (16, 64, 64)
    back = arr2[0:32].read()
    assert (np.abs(back.astype(float) - img) / CAM.sigma(img)).max() <= 0.5 + 1e-6
    with pytest.raises(ValueError):
        arr2[3:19].write(img[3:19])
    with pytest.raises(ValueError):
        arr2[0:16, 0:100].write(img[0:16, 0:100])
    with pytest.raises(TypeError):
        arr2[0:16].write(img[0:16].astype(np.uint8))
    arr2.close()


def test_downsample_matches_numpy():
    img = synth((9, 37, 50))
    for f in ((1, 2, 2), (2, 2, 2), (1, 3, 2)):
        out = np.empty(tuple(-(-n // k) for n, k in zip(img.shape, f)), np.uint16)
        b4d._b4d.downsample(img, out, 2, *img.shape, *f, 3)
        ref = np.empty_like(out)
        for z in range(out.shape[0]):
            for y in range(out.shape[1]):
                for x in range(out.shape[2]):
                    blk = img[z * f[0]:(z + 1) * f[0], y * f[1]:(y + 1) * f[1], x * f[2]:(x + 1) * f[2]].astype(np.uint32)
                    ref[z, y, x] = (blk.sum() + blk.size // 2) // blk.size
        assert np.array_equal(out, ref), f


def test_multiscale_group(tmp_path):
    img = synth((32, 200, 260))
    path = str(tmp_path / "ms.zarr")
    ms = zs.create_multiscale(path, img.shape, dtype="uint16", shards=(16, 128, 128), chunks=(16, 32, 32),
                              codec=B4D(q=1, mode=2, camera=CAM, predictor="above", entropy="rice"), levels=2, factors=(2, 2, 2), threads=2, writers=2)
    assert ms.nlevels == 3
    for z0 in range(0, 32, 16):
        ms[z0:z0 + 16].write(img[z0:z0 + 16])
    ms.close()
    g = zarr.open_group(path, mode="r")
    assert [g[str(k)].shape for k in range(3)] == [(32, 200, 260), (16, 100, 130), (8, 50, 65)]
    assert [ms.level(k).chunks[0] for k in range(3)] == [16, 16, 8] and [ms.level(k).shards[0] for k in range(3)] == [16, 16, 16]
    assert g.attrs["ome"]["multiscales"][0]["datasets"][2]["coordinateTransformations"][0]["scale"] == [4.0, 4.0, 4.0]
    l0 = g["0"][:]
    assert (np.abs(l0.astype(float) - img) / CAM.sigma(img)).max() <= 0.5 + 1e-6
    l1 = g["1"][:]; ref = np.empty(l1.shape, np.uint16); b4d._b4d.downsample(img, ref, 2, *img.shape, 2, 2, 2, 2)
    # level 1 is quantised against its own noise model (8 averaged pixels: conv/8, rn*sqrt(8)): bound q/2 of that sigma
    cam1 = b4d.Camera(CAM.conversion / 8, CAM.offset, CAM.read_noise * np.sqrt(8), exact_vst=False)
    # bound q/2 sigma in the transform domain plus the half-DN rounding of the integer output (sigma is only ~3 DN here)
    assert ((np.abs(l1.astype(float) - ref) - 0.5).clip(0) / cam1.sigma(ref)).max() <= 0.5 + 1e-6
    md1 = ms.level(1).inner[0]["configuration"]; assert abs(md1["conversion"] - CAM.conversion / 8) < 1e-9
    ms2 = zs.open_multiscale(path); assert ms2.nlevels == 3 and ms2.factors == (2, 2, 2); ms2.close()
    with pytest.raises(ValueError):                                   # odd shard depth with z factor 2
        zs.create_multiscale(str(tmp_path / "bad"), img.shape, dtype="uint16", shards=(9, 128, 128), chunks=(3, 32, 32), codec="raw", levels=1, factors=(2, 2, 2))
    with pytest.raises(ValueError):                                   # levels is required
        zs.create_multiscale(str(tmp_path / "bad2"), img.shape, dtype="uint16", shards=(16, 128, 128), chunks=(16, 32, 32), codec="raw")
    assert zs.auto_levels((4096, 10640, 14192), (1, 2, 2)) == 5


def test_ome_ngff_valid_and_pyramid_codec(tmp_path):
    ome_zarr = pytest.importorskip("ome_zarr")
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader
    img = synth((16, 130, 200)); path = str(tmp_path / "ome.zarr")
    ms = zs.create_multiscale(path, img.shape, dtype="uint16", shards=(16, 128, 128), chunks=(8, 64, 64),
                              codec=B4D(q=1, mode=2, camera=CAM, predictor="above", entropy="rice"), levels=1, factors=(2, 2, 2),
                              scale=(2.0, 0.5, 0.5), units="micrometer", threads=2, pyramid_codec=numcodecs.Zstd(level=3))
    ms[0:16].write(img); ms.close()
    node = list(Reader(parse_url(path, mode="r"))())[0]
    assert "Multiscales" in [type(s).__name__ for s in node.specs]
    assert [d.shape for d in node.data] == [(16, 130, 200), (8, 65, 100)]
    assert node.metadata["coordinateTransformations"][1][0]["scale"] == [4.0, 1.0, 1.0]
    import json
    md1 = json.load(open(f"{path}/1/zarr.json"))
    assert md1["dimension_names"] == ["z", "y", "x"] and [c["name"] for c in md1["codecs"][0]["configuration"]["codecs"]] == ["bytes", "zstd"]
    ref = np.empty((8, 65, 100), np.uint16); b4d._b4d.downsample(img, ref, 2, *img.shape, 2, 2, 2, 2)
    assert np.array_equal(zarr.open_array(f"{path}/1", mode="r")[:], ref)      # zstd level is lossless


def test_multiscale_accumulates_across_blocks(tmp_path):
    """level k is written once 2**k blocks have arrived (constant depth), remainder flushed at close"""
    img = synth((40, 100, 120)); path = str(tmp_path / "acc.zarr")
    ms = zs.create_multiscale(path, img.shape, dtype="uint16", shards=(8, 64, 64), chunks=(8, 32, 32), codec="raw", levels=2, factors=(2, 2, 2), threads=2)
    for z0 in range(0, 40, 8):
        ms[z0:z0 + 8].write(img[z0:z0 + 8])
    ms.close()
    g = zarr.open_group(path, mode="r")
    for k, f in ((1, 2), (2, 4)):
        ref = np.empty(tuple(-(-n // f) for n in img.shape), np.uint16)
        tmp = img
        for _ in range(k):
            o = np.empty(tuple(-(-n // 2) for n in tmp.shape), np.uint16); b4d._b4d.downsample(tmp, o, 2, *tmp.shape, 2, 2, 2, 2); tmp = o
        assert np.array_equal(g[str(k)][:], tmp), k
    assert ms.level(2).shards[0] == 8 and ms.level(2).chunks[0] == 8


@pytest.mark.parametrize("Z,bd,levels,writers", [(64, 4, 3, 3), (70, 8, 4, 2), (33, 16, 3, 2), (64, 16, 4, 1)])
def test_multiscale_cascade_exact(tmp_path, Z, bd, levels, writers):
    """pyramid built block by block (any block depth, concurrent writers) equals the pyramid of the whole array"""
    Y, X = 40, 48
    img = (np.arange(Z * Y * X, dtype=np.uint32) * 7919 % 60000).astype(np.uint16).reshape(Z, Y, X)
    path = str(tmp_path / "c.zarr")
    ms = zs.create_multiscale(path, img.shape, dtype="uint16", shards=(bd, 32, 32), chunks=(min(bd, 4), 16, 16), codec="raw",
                              levels=levels, factors=(2, 2, 2), threads=2, writers=writers)
    for z0 in range(0, Z, bd):
        ms[z0:min(z0 + bd, Z)].write(img[z0:min(z0 + bd, Z)])
    ms.close()
    g = zarr.open_group(path, mode="r"); tmp = img
    for k in range(1, levels + 1):
        o = np.empty(tuple(-(-s // 2) for s in tmp.shape), np.uint16); b4d._b4d.downsample(tmp, o, 2, *tmp.shape, 2, 2, 2, 2); tmp = o
        assert np.array_equal(g[str(k)][:], tmp), k


def test_multiscale_early_stop_flush(tmp_path):
    """acquisition that stops before the array end (e.g. after drops): partial accumulators flush as padded shards"""
    Z = 104; img = (np.arange(Z * 40 * 48, dtype=np.uint32) * 7919 % 60000).astype(np.uint16).reshape(Z, 40, 48)
    path = str(tmp_path / "s.zarr")
    ms = zs.create_multiscale(path, (160, 40, 48), dtype="uint16", shards=(8, 32, 32), chunks=(4, 16, 16), codec="raw", levels=3, factors=(2, 2, 2), threads=2, writers=3)
    for z0 in range(0, Z, 8):
        ms[z0:z0 + 8].write(img[z0:z0 + 8])
    ms.close()
    g = zarr.open_group(path, mode="r"); tmp = img
    for k in range(1, 4):
        o = np.empty(tuple(-(-s // 2) for s in tmp.shape), np.uint16); b4d._b4d.downsample(tmp, o, 2, *tmp.shape, 2, 2, 2, 2); tmp = o
        assert np.array_equal(g[str(k)][:tmp.shape[0]], tmp), k


def test_fused_level1_matches_downsampler(tmp_path):
    img = synth((32, 133, 171))
    arr = zs.create(str(tmp_path / "f.zarr"), img.shape, dtype="uint16", shards=(16, 128, 128), chunks=(16, 64, 64), codec="raw", threads=3, sink="memory")
    ds = np.zeros((8, 67, 86), np.uint16); arr._write(img[:16], (0, 0, 0), None, ds_out=ds, factors=(2, 2, 2))
    ref = np.empty_like(ds); b4d._b4d.downsample(img[:16], ref, 2, 16, 133, 171, 2, 2, 2, 2)
    assert np.array_equal(ds, ref)
    ds = np.zeros((16, 67, 86), np.uint16); arr._write(img[:16], (0, 0, 0), None, ds_out=ds, factors=(1, 2, 2))
    ref = np.empty_like(ds); b4d._b4d.downsample(img[:16], ref, 2, 16, 133, 171, 1, 2, 2, 2)
    assert np.array_equal(ds, ref)
    arr.close()


def test_direct_io_roundtrip_and_exact_lengths(tmp_path):
    import os
    img = synth((40, 300, 700)); path = str(tmp_path / "d.zarr")
    arr = zs.create(path, img.shape, dtype="uint16", shards=(16, 128, 256), chunks=(16, 64, 64), codec=B4D(q=1, mode=2, camera=CAM, predictor="left_above", entropy="huffman"), threads=3, direct_io=True)
    for z0 in range(0, 40, 16):
        arr[z0:min(z0 + 16, 40)].write(img[z0:min(z0 + 16, 40)])
    arr.close()
    back = zarr.open_array(path, mode="r")[:]
    assert (np.abs(back.astype(float) - img) / CAM.sigma(img)).max() <= 0.5 + 1e-6
    assert arr.stats["direct_used"] + arr.stats["direct_fallback"] > 0
    # reference lengths from the buffered path
    ref = str(tmp_path / "b.zarr")
    arr2 = zs.create(ref, img.shape, dtype="uint16", shards=(16, 128, 256), chunks=(16, 64, 64), codec=B4D(q=1, mode=2, camera=CAM, predictor="left_above", entropy="huffman"), threads=3, direct_io=False)
    for z0 in range(0, 40, 16):
        arr2[z0:min(z0 + 16, 40)].write(img[z0:min(z0 + 16, 40)])
    arr2.close()
    for dp, _, fs in os.walk(path):
        for f in fs:
            if f.endswith(".json"): continue
            rel = os.path.relpath(os.path.join(dp, f), path)
            assert open(os.path.join(dp, f), "rb").read() == open(os.path.join(ref, rel), "rb").read(), rel


def test_cuda_kernel_emulation_identical(tmp_path):
    """the CUDA Rice kernel (run through its CPU emulation) produces the same store as the C writer"""
    import shutil, subprocess
    if shutil.which("gcc") is None:
        pytest.skip("gcc needed for the kernel emulation")
    from b4d.cuda_encoder import CudaTileEncoder
    img = synth((40, 300, 420)); img[3, 10:20, :] = 100; img[:, 0:64, 0:64] = 0; img[:, 64:128, 0:64] = 0x2121
    def write(path, enc):
        arr = zs.create(path, img.shape, dtype="uint16", shards=(16, 128, 256), chunks=(16, 64, 64), codec=B4D(q=2, mode=2, camera=CAM, predictor="above", entropy="rice"), threads=2)
        if enc:
            arr.encoder = enc.encode_tiles
        for z0 in range(0, 40, 16):
            arr[z0:min(z0 + 16, 40)].write(img[z0:min(z0 + 16, 40)])
        arr.close()
    import os
    write(str(tmp_path / "ref"), None); write(str(tmp_path / "emu"), CudaTileEncoder(CAM, q=2, mode=2, predictor="above", backend="emu", batch_tiles=32))
    for dp, _, fs in os.walk(str(tmp_path / "ref")):
        for f in fs:
            if f.endswith(".json"): continue
            rel = os.path.relpath(os.path.join(dp, f), str(tmp_path / "ref"))
            assert open(os.path.join(dp, f), "rb").read() == open(str(tmp_path / "emu" / rel), "rb").read(), rel


def test_fused_level1_matches_downsampler_uint8(tmp_path):
    """8-bit cameras use the fused level-1 path too: same box mean and rounding as the separate pass"""
    img = (synth((32, 133, 171)) % 256).astype(np.uint8)
    arr = zs.create(str(tmp_path / "f8.zarr"), img.shape, dtype="uint8", shards=(16, 128, 128), chunks=(16, 64, 64), codec="raw", threads=3, sink="memory")
    for fac, nz in (((2, 2, 2), 8), ((1, 2, 2), 16)):
        ds = np.zeros((nz, 67, 86), np.uint8); arr._write(img[:16], (0, 0, 0), None, ds_out=ds, factors=fac)
        ref = np.empty_like(ds); b4d._b4d.downsample(img[:16], ref, 1, 16, 133, 171, *fac, 2)
        assert np.array_equal(ds, ref), fac
    arr.close()


def test_multiscale_uint8_fused_pyramid_exact(tmp_path):
    """a uint8 multiscale write takes the fused path and its pyramid equals the separate downsampler's"""
    Z, bd = 64, 8
    img = (np.arange(Z * 40 * 48, dtype=np.uint32) * 7919 % 251).astype(np.uint8).reshape(Z, 40, 48)
    path = str(tmp_path / "u8.zarr")
    ms = zs.create_multiscale(path, img.shape, dtype="uint8", shards=(bd, 32, 32), chunks=(4, 16, 16), codec="raw", levels=3, factors=(2, 2, 2), threads=2, writers=2)
    for z0 in range(0, Z, bd):
        ms[z0:z0 + bd].write(img[z0:z0 + bd])
    ms.close()
    assert ms.stats.get("fused_blocks", 0) > 0
    g = zarr.open_group(path, mode="r"); tmp = img
    for k in range(1, 4):
        o = np.empty(tuple(-(-s // 2) for s in tmp.shape), np.uint8); b4d._b4d.downsample(tmp, o, 1, *tmp.shape, 2, 2, 2, 2); tmp = o
        assert np.array_equal(g[str(k)][:], tmp), k


def test_prefaulted_zeros():
    a = zs._prefaulted_zeros((3, 1000, 1000), np.uint16, threads=4)
    assert a.shape == (3, 1000, 1000) and a.dtype == np.uint16 and not a.any()
    assert zs._prefaulted_zeros((0, 5), np.uint8).size == 0


@pytest.mark.parametrize("writers", [1, 3, 6])
def test_fused_pyramid_reservation_is_race_free(tmp_path, writers):
    """level-1 planes written by the compression workers straight into the accumulator (fused path) must never
    overwrite planes that have not been written out yet, whatever the number of writers in flight"""
    Z, bd = 160, 8
    img = (np.arange(Z * 40 * 48, dtype=np.uint32) * 7919 % 60000).astype(np.uint16).reshape(Z, 40, 48)
    path = str(tmp_path / "rc.zarr")
    ms = zs.create_multiscale(path, (Z, 40, 48), dtype="uint16", shards=(8, 32, 32), chunks=(4, 16, 16), codec="raw", levels=3, factors=(2, 2, 2), threads=2, writers=writers)
    for z0 in range(0, Z, bd):
        ms[z0:z0 + bd].write(img[z0:z0 + bd])
    ms.close()
    assert ms.stats.get("fused_blocks", 0) > 0
    g = zarr.open_group(path, mode="r"); tmp = img
    for k in range(1, 4):
        o = np.empty(tuple(-(-s // 2) for s in tmp.shape), np.uint16); b4d._b4d.downsample(tmp, o, 2, *tmp.shape, 2, 2, 2, 2); tmp = o
        assert np.array_equal(g[str(k)][:], tmp), k


def test_multiscale_with_leading_channel_axis(tmp_path):
    """OME (c, z, y, x) arrays: ms[c, z0:z1].write() writes channel c at every level (chunk keys c/z/y/x)."""
    import json
    from concurrent.futures import ThreadPoolExecutor

    zarr = pytest.importorskip("zarr")
    import b4d.zarr3  # noqa: F401

    root = tmp_path / "ms.zarr"
    nz = 16  # two channels (c extent 2, chunk and shard extent 1 on c): one MultiscaleArray instance per channel
    arrays = []
    for k in range(2):
        f = 2**k
        # the level codec as create_multiscale / voxel author it: level k averages 8**k pixels -> conversion / 8**k, rn * sqrt(8**k)
        codec = dict(q=1.0, mode=2, conversion=2.0 / 8**k, offset=100.0, read_noise=3.0 * np.sqrt(8**k), predictor="above", entropy="rice")
        md = zs._metadata((2, nz // f, 64 // f, 96 // f), "uint16", (1, 8 // f, 64 // f, 96 // f), (1, 8 // f, 32 // f, 32 // f),
                          zs.inner_codecs_for(codec), dimension_names=["c", "z", "y", "x"])
        zs._write_metadata(str(root / str(k)), md)
        arrays.append(zs.ZarrArray(str(root / str(k)), md, 4, True, 1))
    (root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
    ms = zs.MultiscaleArray(str(root), arrays, (2, 2, 2), 4)
    ms._exec = ThreadPoolExecutor(1)
    written = []
    ms.on_written = lambda level, z0, z1: written.append((level, z0, z1))  # a staging host uploads exactly these
    rng = np.random.default_rng(3)
    data = (rng.normal(500, 30, (nz, 64, 96))).astype(np.uint16)
    ms[0, 0:8].write(data[:8]).result()
    ms[0, 8:16].write(data[8:16]).result()
    with pytest.raises(ValueError):
        ms[1, 0:8]  # a second channel needs its own instance
    ms.close()
    assert sorted(written) == [(0, 0, 8), (0, 8, 16), (1, 0, 4), (1, 4, 8)]  # level 0 per block, level 1 per shard depth
    assert (root / "0" / "c" / "0" / "0" / "0" / "0").exists() and (root / "1" / "c" / "0" / "0" / "0" / "0").exists()
    l0 = zarr.open(str(root / "0"), mode="r")[0]
    l1 = zarr.open(str(root / "1"), mode="r")[0]
    sigma = np.sqrt(2.0 * 400 + (3.0 * 2.0) ** 2)  # noise at ~500 DN with conversion 2 DN/e-, offset 100, rn 3 e-
    assert l0.shape == (nz, 64, 96) and np.abs(l0.astype(np.int32) - data).max() <= 0.6 * sigma + 1  # q = 1: |err| ~ sigma/2 (+ rounding)
    mean = data.reshape(8, 2, 32, 2, 48, 2).mean(axis=(1, 3, 5))
    assert np.abs(l1.astype(np.float64) - mean).max() <= 0.6 * sigma / np.sqrt(8) + 1.5

