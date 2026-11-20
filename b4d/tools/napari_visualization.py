"""Open a B4D-compressed zarr v3 array or OME-Zarr multiscale group in napari.

    python napari_visualization.py                       # file dialog
    python napari_visualization.py C:\\b4d_out\\stream.zarr   # or edit DATASET below

Requirements (in the b4d environment):
    pip install napari[all] napari-ome-zarr dask

What this does:
  * imports b4d.zarr3 so zarr-python can decode the "b4d" codec (this is the only thing a viewer needs);
  * opens either a plain array (zarr.json with node_type "array") or an OME-NGFF multiscale group
    (zarr.json with node_type "group" and ome.multiscales), as written by b4d.zarr_store;
  * wraps each level in a lazy dask array so only the chunks on screen are decoded, and hands the list of
    levels to napari as one multiscale image (napari picks the level that fits the current zoom);
  * reads the physical voxel size from the OME metadata for the scale bar / anisotropy.
"""
import sys
from pathlib import Path

import numpy as np

# ---- EDIT THIS to point at a dataset, or leave None to get a dialog / use the command line argument
DATASET = None   # e.g. r"C:\b4d_out\stream.zarr"

# default contrast window as fraction of the data range; set to None to let napari auto-contrast
CONTRAST_PERCENTILES = (0.5, 99.8)


def choose_dataset():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    if DATASET:
        return Path(DATASET)
    try:
        from qtpy.QtWidgets import QApplication, QFileDialog
        QApplication.instance() or QApplication(sys.argv)  # keep a Qt app alive for the dialog
        path = QFileDialog.getExistingDirectory(None, "Select a .zarr dataset (the folder containing zarr.json)")
        if not path:
            sys.exit("no dataset selected")
        return Path(path)
    except ImportError:
        sys.exit("pass the dataset path as an argument or set DATASET in the script")


def open_levels(path: Path):
    """Return (list of lazy level arrays, largest first), scale (per axis), axis names."""
    import json
    import dask.array as da
    import zarr
    import b4d.zarr3  # noqa: F401  registers the b4d codec with zarr-python

    meta = json.loads((path / "zarr.json").read_text())
    if meta.get("node_type") == "array":
        z = zarr.open_array(str(path), mode="r")
        return [da.from_zarr(z)], (1.0,) * z.ndim, None
    ome = meta.get("attributes", {}).get("ome") or meta.get("attributes", {})
    ms = ome["multiscales"][0]
    axes = [a["name"] for a in ms["axes"]]
    levels, scales = [], []
    for ds in ms["datasets"]:
        z = zarr.open_array(str(path / ds["path"]), mode="r")
        levels.append(da.from_zarr(z))
        sc = next((t["scale"] for t in ds.get("coordinateTransformations", []) if t["type"] == "scale"), [1.0] * z.ndim)
        scales.append(tuple(float(v) for v in sc))
    return levels, scales[0], axes


def main():
    path = choose_dataset()
    if not (path / "zarr.json").exists():
        sys.exit(f"{path} does not contain zarr.json")
    levels, scale, axes = open_levels(path)
    print(f"{path}: {len(levels)} level(s); level 0 shape {levels[0].shape} dtype {levels[0].dtype}; voxel size {scale}")
    for k, lv in enumerate(levels):
        print(f"  level {k}: {lv.shape}  chunks {lv.chunksize}")

    contrast = None
    if CONTRAST_PERCENTILES:
        # sample the coarsest level to set a sensible window without decoding full resolution
        sample = np.asarray(levels[-1][tuple(slice(0, min(s, 64)) for s in levels[-1].shape)])
        lo, hi = np.percentile(sample, CONTRAST_PERCENTILES)
        contrast = (float(lo), float(max(hi, lo + 1)))

    import napari
    viewer = napari.Viewer(title=f"b4d: {path.name}")
    viewer.add_image(levels if len(levels) > 1 else levels[0], name=path.name, multiscale=len(levels) > 1,
                     scale=scale, contrast_limits=contrast, colormap="gray", blending="additive")
    if axes:
        viewer.dims.axis_labels = axes
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()


if __name__ == "__main__":
    main()
