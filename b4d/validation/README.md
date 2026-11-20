# b4d/validation — codec consistency against the frozen reference

`validation/` holds pass/fail checks of correctness, not measurements (those are in `tools/`).

`compare_codecs.py` produces the numbers in `docs/COMPARISON.md` from a set of camera chunks.

`test_codec_consistency.py` (pytest) validates the five codec options, `above`/`mean3d` x `rice`/`ans` and `b3d`, in
lossless mode and in modes 1 and 2 at q = 1 and 2 on a camera-model phantom:

| check | reference |
|---|---|
| C encoder bytes (sha256 + length) | `golden_bitstreams.json`, recorded with the build whose `above`/`rice` output `tests/test_mean3d.py` pins to the writer-tuning-campaign build of 2026-10-05 (the validated production bitstream) |
| decoder inverts the encoder | exact for mode 0; within q/2 sigma + the half-ADU rounding otherwise |
| `compress_tiles` (the writer's path) | byte-identical to `compress` of each tile |
| native sharded-zarr writer (`b4d_zarr.c`) | the shard file is exactly the tiles + index; zarr-python decodes it to the C decoder's array |
| CUDA encoder (needs a GPU) | byte-identical to the C encoder for the four native options |
| `b3d` (fifth option, `test_b3d_reference.py`) | byte-identical to the original B3D implementation: the 24 chunks of its demo file and its win64 GPU binary driven live (70 cases) |

```
cd b4d
python -m pytest validation -q
```

The hashes were re-recorded on 2026-10-12 when the container magic became `B4DP`: each new blob was checked to equal the
previously validated bytes everywhere except those four magic bytes.

`test_b3d_reference.py` fetches the original repository's demo file (and, on Windows with an NVIDIA GPU, its win64
binary) into `%LOCALAPPDATA%/b4d/validation` with pinned sha256; set `B4D_ORIGINAL_REPO` to a local clone to skip
the download. Those tests skip when the material is unavailable.

For the four native options the original B3D implementation (Balazs et al. 2017, CUDA) is not a byte reference: they
change the transform, the predictor and the entropy stage (see `docs/CHANGES_FROM_B3D.md`); their byte
reference is the production bitstream frozen in `tests/test_mean3d.py` and extended here. The fifth option *is* the
original pipeline, and `test_b3d_reference.py` compares it byte for byte with the original implementation.

Re-recording the golden file is a format change: `B4D_UPDATE_GOLDEN=1 python -m pytest validation -q -k frozen`,
then commit the JSON together with the format-version bump and the reason.

Unit tests (fast, run on every change): `python -m pytest tests -q`; the C unit tests: `make test` (`tests/c/b4d_test.c`).

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).
