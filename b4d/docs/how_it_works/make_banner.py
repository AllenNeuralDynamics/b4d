"""Banner for the top of the README: neurites of a public exaSPIM recording turning, from left to right, into what
b4d stores.

    python make_banner.py            -> figs/banner.png (1600 x 160 pixels)

Data: dataset 859684_2026-10-06_16-58-46 (the camera of Section 1.2), channel 488 nm, tile 13, level 1 (2 µm in z,
1.496 µm in y and x), a 128-plane slab around z = 8,570, y = 5,112, x = 9,146 (level-0 pixels). Downloaded once
(anonymous HTTPS); the projection and the middle plane are kept in banner_data.npz.

Left: maximum-intensity projection of the slab. Middle: the residual of the slab's middle plane after variance
stabilization with the camera model, quantization at q = 1σ and prediction from the three pixels of the preceding row
(mostly 0, mid-gray). Right: the Rice code of those residuals, one bit per pixel. The parts dissolve into each other
pixel by pixel. Grayscale, no text.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_example_chunk as fx

LEVEL = 1
CENTRE0 = (8570, 5112, 9146)              # z, y, x at level 0 (the neuroglancer view)
NZ, NY, NX = 128, 160, 1600               # slab at level 1; the banner is NX x NY pixels
G, O, RN = 1.373, 34.09, 2.65             # camera model: ADU/e⁻, ADU, e⁻
CACHE = os.path.join(HERE, "banner_data.npz")


def slab():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["mip"], d["mid"]
    fx.ROOT = fx.ROOT.rsplit("/", 1)[0] + f"/{LEVEL}"
    c = [v // 2 ** LEVEL for v in CENTRE0]
    lo = [c[0] - NZ // 2, c[1] - NY // 2, c[2] - NX // 2]
    a = fx.fetch_box(lo, (NZ, NY, NX))
    np.savez_compressed(CACHE, mip=a.max(axis=0), mid=a[NZ // 2])
    return a.max(axis=0), a[NZ // 2]


def residuals(adu):
    N = (adu.astype(float) - O) / G                                  # photoelectrons
    Q = np.rint(2 * np.sqrt(np.clip(N + 3 / 8 + RN ** 2, 0, None)))  # stabilized, quantized at q = 1σ
    up = np.pad(Q[:-1], ((0, 0), (1, 1)), mode="edge")
    R = np.zeros_like(Q)
    R[1:] = Q[1:] - np.rint((up[:, :-2] + up[:, 1:-1] + up[:, 2:]) / 3)
    return R


def rice_rows(R, k=1):
    out = np.zeros(R.shape)
    for i, row in enumerate(R.astype(int)):
        u = np.where(row >= 0, 2 * row, -2 * row - 1)
        bits = []
        for v in u:
            bits += [1] * (v >> k) + [0] + [(v >> b) & 1 for b in range(k - 1, -1, -1)]
            if len(bits) >= R.shape[1]:
                break
        out[i] = bits[:R.shape[1]]
    return out


def main():
    mip, mid = slab()
    mip = mip.astype(float)
    lo, hi = np.percentile(mip, (2, 99.8))
    raw = np.clip((mip - lo) / (hi - lo), 0, 1) ** 0.7
    R = residuals(mid)
    res = np.clip(0.5 + R / 8, 0, 1)
    bits = rice_rows(R)

    rng = np.random.default_rng(1)
    x = (np.arange(NX) + 0.5) / NX
    u = rng.uniform(size=(NY, NX))
    s = lambda c, w: np.clip((x - c) / (2 * w) + 0.5, 0, 1) ** 1.5   # step confined to [c − w, c + w]
    img = np.where(u < s(0.42, 0.12), res, raw)                      # projection -> residual
    img = np.where(u < s(0.74, 0.12), bits, img)                     # residual -> bits

    out = os.path.join(HERE, "figs", "banner.png")
    plt.imsave(out, img, cmap="gray", vmin=0, vmax=1)
    print(out, img.shape)


if __name__ == "__main__":
    main()
