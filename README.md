# B4D

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](b4d/pyproject.toml) [![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#install) [![CUDA optional](https://img.shields.io/badge/CUDA-optional-76b900)](b4d/README.md#gpu-encoder) [![OME-Zarr v3](https://img.shields.io/badge/OME--Zarr-v3-orange)](b4d/README.md#native-zarr-writer) [![Cite](https://img.shields.io/badge/cite-CITATION.cff-informational)](CITATION.cff)

B4D compresses images from scientific cameras within the noise limit: the information it discards lies inside the
noise the camera itself records, so every reconstructed pixel stays within half a standard deviation of its own noise.
At that bound B4D reaches the compression ratios of lossy codecs (6- to 9-fold on recorded data) at the speed of
lossless codecs (about 800 MB/s per core, 80 GB/s on a 128-core CPU, 22 GB/s per GPU). This makes it uniquely suited
to compression at acquisition, on data streaming from cameras at full rate, with no anticipated impact on downstream
analysis.

This repository holds the codec and the acquisition stack needed to use it at the speed of state-of-the-art
high-speed scientific cameras: a native sharded zarr v3 writer with OME metadata, CPU and GPU (CUDA) encoders with
identical output, and zero-copy example drivers for scientific cameras.

B4D builds on B3D (Balázs et al. 2017, [github.com/balintbalazs/B3D](https://github.com/balintbalazs/B3D)). It keeps
B3D's idea and changes the transform, predictor, entropy coder and container;
[`b4d/docs/CHANGES_FROM_B3D.md`](b4d/docs/CHANGES_FROM_B3D.md) lists the differences. The original B3D pipeline is also
included as a selectable option, validated byte for byte against the original implementation.

## Packages

Five independently installable packages, layered from the camera up: b4d and four example camera drivers.

```
egrabber/     Euresys eGrabber (CoaXPress frame grabbers): EGrabberSource produces zero-copy frame blocks
dcam/         Hamamatsu DCAM, pvcam/ Teledyne PVCAM, tucam/ Tucsen TUCam: block sources of the same contract
              (none of the drivers depends on b4d)
   │  ── BlockSource contract (b4d/src/b4d/ingest/protocol.py): blocks of frames in pinned memory ──
b4d/          the codec (C, Blosc2), native sharded-zarr writer, CUDA encoder, b4d.ingest (BlockRecorder,
              SimulatedCameraSource), simulator and tuning tools
```

| directory | package | needs | command-line tools | tested on |
|---|---|---|---|---|
| [`b4d/`](b4d/README.md) | `b4d` | Python ≥ 3.10, a C compiler, c-blosc2 | `b4d-record`, `b4d-tune` | |
| [`egrabber/`](egrabber/README.md) | `egrabber_camera` | Windows, the Euresys eGrabber SDK, clang-cl | `egrabber-grab` | Vieworks VPS-245MX2 on a Euresys Coaxlink Quad CXP-12 |
| [`dcam/`](dcam/README.md) | `dcam_camera` | the Hamamatsu DCAM-API | | Hamamatsu ORCA-Flash4.0 V3 |
| [`pvcam/`](pvcam/README.md) | `pvcam_camera` | the Teledyne PVCAM SDK | | simulated PVCAM runtime |
| [`tucam/`](tucam/README.md) | `tucam_camera` | the Tucsen TUCam SDK | | Tucsen Aries 6504 |

b4d has `src/`, `tests/` (pytest), `validation/` (pass/fail checks against a reference), `tools/` (simulators,
benchmarks, studies), `examples/` and `docs/`; egrabber has `src/`, `tests/`, `validation/`, `tools/` (camera
utilities) and `examples/`; dcam, pvcam and tucam have `src/`, `tests/` and `examples/`.
Each has a `pyproject.toml` and the MIT `LICENSE`.

## Install

`b4d` is the only package needed to compress data or to try the writer with the simulated camera.

macOS:

```bash
brew install c-blosc2 pkg-config
pip install "./b4d[zarr,sim]"
```

Linux: install `libblosc2-dev` and `pkg-config` from your package manager, then the same `pip` line.

Windows, inside a conda environment (clang-cl encodes 13–19 % faster than MSVC, with identical output):

```
conda install -c conda-forge c-blosc2 clang
set B4D_CC=clang-cl
set B4D_CFLAGS=/clang:-O3 /clang:-mtune=native
pip install "./b4d[zarr,sim]"
```

Other extras: `gpu` (CuPy, for the CUDA encoder), `viewers` (napari), `dev` (test and lint tools). `b4d/README.md` covers
the remaining build options.

Each camera driver has its own prerequisites; see its README (for example [`egrabber/README.md`](egrabber/README.md)).

## Quick start

Compress an array:

```python
import numpy as np
import b4d

cam = b4d.Camera(conversion=2.0,    # DN per electron
                 offset=100.0,      # baseline in DN
                 read_noise=3.0)    # electrons RMS

blob = b4d.compress(img, camera=cam)                      # within noise level: q=1, mode=1, above3, Rice
blob = b4d.compress(img, q=0)                             # lossless
out = b4d.decompress(blob)                                # same shape and dtype
```

Record from a simulated camera into a sharded zarr, with no hardware:

```bash
b4d-record --source simulated --source-arg shape=2048,2048 --source-arg fps=100 --seconds 10 --sink memory
```

Record from a Euresys camera:

```
b4d-record --source egrabber_camera.source:EGrabberSource --source-arg serial=<camera serial> --seconds 60 --q 1 --out <directory>
```

Every recording ends with a machine-readable `RESULT {...}` line (frame rate, frames lost, compression ratio, timings).
`b4d-tune` finds the best writer settings for a camera on the machine it runs on.

## The interface between a camera and the writer

A *source* owns a pool of large block buffers (`nblocks × frames_per_block` frames) that the camera's DMA engine
fills. The writer compresses a completed block in place and releases it, so nothing is copied. The contract is
structural: a vendor package implements `info`, `frames_per_block`, `nblocks`, `start`, `wait_block`, `release`,
`stop`, `stats` and `close` (optionally `block_array`) and never imports b4d. b4d's `BlockRecorder` accepts any such
object. To support another camera, copy `egrabber/` as a template, or start from
`b4d/examples/minimal_block_source.py`, and change nothing in b4d.

```python
from b4d.ingest import BlockRecorder, RecorderConfig, SimulatedCameraSource   # or: from egrabber_camera import EGrabberSource

src = SimulatedCameraSource(shape=(2048, 2048), fps=100, frames_per_block=64, nblocks=3)
with BlockRecorder(src, RecorderConfig(path="run.zarr", q=1, gpu="none")) as rec:
    rec.start(seconds=30)
    print(rec.report(rec.run()))
```

## Tests

```bash
cd b4d
python -m pytest tests validation -q     # unit tests and the frozen-bitstream checks
make SYSTEM_BLOSC=1 test                 # the C suite
cd ..
python -m pytest -q                      # the four example camera drivers (egrabber, dcam, pvcam, tucam), against fake SDKs
```

The compressed bitstream is a format: `validation/` compares the encoder's output with recorded hashes, so any change
to the codec that alters a single byte fails there. Each camera driver has its own `tests/`
(egrabber also `validation/`); the checks that need a camera or a GPU skip when none is present.

## Documentation

- [`b4d/README.md`](b4d/README.md): how b4d works (noise model, encoding, container, measured rates), install, the
  Python API, zarr codecs, the native writer, the GPU encoder and viewers.
- [`b4d/docs/CHANGES_FROM_B3D.md`](b4d/docs/CHANGES_FROM_B3D.md): how the algorithm differs from B3D.
- [`b4d/docs/WRITER_DESIGN.md`](b4d/docs/WRITER_DESIGN.md): design of the codec and the writer, with measurements.
- [`b4d/docs/GPU_VS_CPU.md`](b4d/docs/GPU_VS_CPU.md): CPU and GPU write rates, hardware prices and which to choose.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).

## Authors

Adam Glaser and Claude (Anthropic's AI model), who wrote the code together.

## License and attribution

MIT; see `LICENSE`.

B4D is derived from the B3D algorithm of Balázs, Deschamps, Albert, Ries and Hufnagel, "A real-time compression
library for microscopy images" (2017).
