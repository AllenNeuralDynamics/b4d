"""Fetch the guide's example chunk: 64 x 64 x 64 pixels of a public AIND exaSPIM dataset (read-only, anonymous HTTPS),
next to a neuroglancer view of 859684_2026-10-06_16-58-46 (exaspim-5x-03, the beta3 camera), channel 488, tile 13,
full resolution. Saves
example_chunk.npy (z, y, x; uint16 raw camera values) next to this file."""
import json, os, struct, sys, urllib.request
import numpy as np
from numcodecs import Zstd

ROOT = "https://aind-open-data.s3.amazonaws.com/859684_2026-10-06_16-58-46/SPIM/tile_000013_ch_488.ome.zarr/0"
CENTRE = (8152, 5333, 10211)        # z, y, x: near a neuroglancer view of the tile (voxel 8166, 5341, 10182; z 13027.5 um, y 13899.5, x 26893.7 x 0.748 um)
# moved by (-14, -8, +29) so that planes 0, 1, 31-33 and 62-63 (the ones drawn in fig02) all hold neurites: the view's own
# chunk had empty planes 0-1 (found by a search over a 256^3 box around the view)
N = 64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_chunk.npy")


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}"} if rng else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_box(lo, shape):
    """Read the box lo .. lo + shape (z, y, x voxels) of the tile at full resolution."""
    meta = json.loads(get(ROOT + "/zarr.json"))
    shard = meta["chunk_grid"]["configuration"]["chunk_shape"][2:]
    sh = meta["codecs"][0]["configuration"]; inner = sh["chunk_shape"][2:]
    assert [c["name"] for c in sh["codecs"]] == ["transpose", "bytes", "zstd"] and sh["codecs"][0]["configuration"]["order"] == [0, 1, 2, 3, 4]
    n_in = [s // i for s, i in zip(shard, inner)]
    out = np.zeros(shape, np.uint16)
    cache = {}
    # every inner chunk the box touches
    rngs = [range(l // i, (l + n - 1) // i + 1) for l, n, i in zip(lo, shape, inner)]
    for gz in rngs[0]:
        for gy in rngs[1]:
            for gx in rngs[2]:
                s_idx = [gz * inner[0] // shard[0], gy * inner[1] // shard[1], gx * inner[2] // shard[2]]
                url = f"{ROOT}/c/0/0/{s_idx[0]}/{s_idx[1]}/{s_idx[2]}"
                if url not in cache:
                    nidx = n_in[0] * n_in[1] * n_in[2]
                    cache[url] = np.frombuffer(get(url, f"-{nidx * 16 + 4}")[:nidx * 16], "<u8").reshape(-1, 2)
                li = [(gz * inner[0] - s_idx[0] * shard[0]) // inner[0], (gy * inner[1] - s_idx[1] * shard[1]) // inner[1],
                      (gx * inner[2] - s_idx[2] * shard[2]) // inner[2]]
                off, nb = cache[url][(li[0] * n_in[1] + li[1]) * n_in[2] + li[2]]
                raw = Zstd().decode(get(url, f"{off}-{off + nb - 1}"))
                a = np.frombuffer(raw, "<u2").reshape(inner)
                o = [gz * inner[0], gy * inner[1], gx * inner[2]]
                sl_src, sl_dst = [], []
                for k in range(3):
                    a0, a1 = max(lo[k], o[k]), min(lo[k] + shape[k], o[k] + inner[k])
                    sl_src.append(slice(a0 - o[k], a1 - o[k])); sl_dst.append(slice(a0 - lo[k], a1 - lo[k]))
                out[tuple(sl_dst)] = a[tuple(sl_src)]
                print("inner chunk", (gz, gy, gx), "bytes", nb, file=sys.stderr)
    return out


def main():
    lo = [c - N // 2 for c in CENTRE]
    out = fetch_box(lo, (N, N, N))
    np.save(OUT, out)
    print("saved", OUT, "origin (z, y, x)", lo, "min/median/max", out.min(), int(np.median(out)), out.max())


if __name__ == "__main__":
    main()
