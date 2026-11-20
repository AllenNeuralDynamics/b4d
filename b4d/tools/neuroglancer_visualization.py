"""Serve a B4D-compressed zarr v3 dataset to neuroglancer (and any other zarr v3 reader) over HTTP.

    python neuroglancer_visualization.py C:\\b4d_out\\stream.zarr                 # serves on http://localhost:8000/
    python neuroglancer_visualization.py C:\\b4d_out\\stream.zarr --codec zstd --port 8010 --open

Viewers such as neuroglancer or viv decode chunks with their own code and do not know the
b4d codec. This server presents a *virtual view* of the dataset that they can read:

  * the same array shape, dtype, dimension names and OME-NGFF multiscale metadata;
  * chunks of the dataset's inner-chunk size, served individually (no sharding codec in the virtual
    metadata, so clients fetch whole chunk objects and need no byte-range or shard-index support);
  * each requested chunk is located in its shard through the shard index, decoded from B4D in C and
    re-encoded as "bytes" (raw little-endian) or "zstd", on demand, with an LRU cache.

Nothing is written to disk and the original dataset is untouched. Decoding a 32x64x64 chunk takes ~1 ms;
the server is threaded, so interactive browsing at any zoom is comfortable on a workstation.

Open in neuroglancer: the script prints a URL for https://neuroglancer-demo.appspot.com with the layer
preconfigured (the browser fetches from localhost directly; `--open` launches it). The source URL is
``zarr3://http://localhost:8000/<name>`` for a multiscale group or an array.

Requirements: the b4d package (for decoding); numcodecs for --codec zstd. No neuroglancer install needed.
"""
import argparse
import functools
import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import zarr

import b4d.zarr3  # noqa: F401  (registers the b4d codec with zarr-python)


class VirtualStore:
    """Virtual, unsharded zarr v3 view of a dataset (plain array or OME-NGFF group of arrays)."""

    def __init__(self, root: Path, codec: str = "bytes", cache_chunks: int = 4096, clevel: int = 1):
        self.root = root
        self.codec = codec
        self.name = root.name
        self.arrays = {}            # relative path ("" or "0", "1", ...) -> zarr array
        self.meta = {}              # relative path -> virtual zarr.json (dict)
        meta = json.loads((root / "zarr.json").read_text())
        if meta.get("node_type") == "array":
            self._add_array("", meta)
        else:
            self.meta[""] = meta  # group: keep attributes (OME multiscales) verbatim
            ome = meta.get("attributes", {}).get("ome") or meta.get("attributes", {})
            for ds in ome["multiscales"][0]["datasets"]:
                p = ds["path"]
                self._add_array(p, json.loads((root / p / "zarr.json").read_text()))
        if codec == "zstd":
            import numcodecs
            self._enc = numcodecs.Zstd(level=clevel)
        self._get_chunk = functools.lru_cache(maxsize=cache_chunks)(self._get_chunk_uncached)

    def _add_array(self, rel, meta):
        self.arrays[rel] = zarr.open_array(str(self.root / rel) if rel else str(self.root), mode="r")
        sh = meta["codecs"][0]
        inner = sh["configuration"]["chunk_shape"] if sh["name"] == "sharding_indexed" else meta["chunk_grid"]["configuration"]["chunk_shape"]
        vcodecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
        if self.codec == "zstd":
            vcodecs.append({"name": "zstd", "configuration": {"level": 1, "checksum": False}})
        v = dict(meta)
        v["chunk_grid"] = {"name": "regular", "configuration": {"chunk_shape": list(inner)}}
        v["chunk_key_encoding"] = {"name": "default", "configuration": {"separator": "/"}}
        v["codecs"] = vcodecs
        v.pop("storage_transformers", None)
        self.meta[rel] = v

    def chunk_shape(self, rel):
        return tuple(self.meta[rel]["chunk_grid"]["configuration"]["chunk_shape"])

    def _get_chunk_uncached(self, rel, idx):
        z = self.arrays[rel]; cs = self.chunk_shape(rel)
        sl = tuple(slice(i * c, min((i + 1) * c, n)) for i, c, n in zip(idx, cs, z.shape))
        if any(s.start >= n for s, n in zip(sl, z.shape)):
            return None
        data = np.asarray(z[sl])                                   # decodes exactly the inner chunk(s) needed
        if data.shape != cs:                                       # edge chunk: pad to the full chunk shape
            full = np.zeros(cs, data.dtype); full[tuple(slice(0, s) for s in data.shape)] = data; data = full
        raw = np.ascontiguousarray(data).astype(data.dtype.newbyteorder("<"), copy=False).tobytes()
        return self._enc.encode(raw) if self.codec == "zstd" else raw

    def get(self, path: str):
        """path relative to the dataset root, e.g. 'zarr.json', '0/zarr.json', '0/c/1/2/3'"""
        parts = [p for p in path.split("/") if p]
        if parts and parts[-1] == "zarr.json":
            rel = "/".join(parts[:-1])
            return ("application/json", json.dumps(self.meta[rel]).encode()) if rel in self.meta else None
        if "c" in parts:
            k = parts.index("c"); rel = "/".join(parts[:k]); idx = tuple(int(p) for p in parts[k + 1:])
            if rel not in self.arrays or len(idx) != len(self.chunk_shape(rel)):
                return None
            blob = self._get_chunk(rel, idx)
            return ("application/octet-stream", blob) if blob is not None else None
        return None


class Handler(BaseHTTPRequestHandler):
    store: VirtualStore = None
    quiet = True

    def _send(self, code, ctype=None, body=b""):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        if ctype:
            self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        prefix = "/" + self.store.name
        if not path.startswith(prefix):
            return self._send(404)
        res = self.store.get(path[len(prefix):])
        if res is None:
            return self._send(404)
        self._send(200, *res)

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


def neuroglancer_url(base_url: str, store: VirtualStore):
    """a neuroglancer-demo URL with the dataset as an image layer"""
    src = f"zarr3://{base_url}/{store.name}"
    is_group = "" not in store.arrays
    state = {"layers": [{"type": "image", "source": src, "name": store.name}], "layout": "4panel"}
    if not is_group:                       # plain array: tell neuroglancer the axis order
        state["layers"][0]["source"] = {"url": src, "transform": {"outputDimensions": {"z": [1, "um"], "y": [1, "um"], "x": [1, "um"]}}}
    return "https://neuroglancer-demo.appspot.com/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")), safe="")


def serve(root: Path, port=8000, host="localhost", codec="bytes", open_browser=False, quiet=True):
    store = VirtualStore(root, codec=codec)
    Handler.store = store; Handler.quiet = quiet
    httpd = ThreadingHTTPServer((host, port), Handler)
    base = f"http://{host}:{port}"
    print(f"serving {root} as a virtual zarr v3 ({codec} chunks) at {base}/{store.name}/")
    for rel, z in store.arrays.items():
        print(f"  {rel or '.'}: shape {z.shape} dtype {z.dtype} virtual chunks {store.chunk_shape(rel)}")
    url = neuroglancer_url(base, store)
    print("\nneuroglancer:\n  " + url + "\n")
    if open_browser:
        webbrowser.open(url)
    return httpd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="path to a .zarr array or OME-Zarr multiscale group")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--codec", default="bytes", choices=["bytes", "zstd"], help="codec of the served chunks")
    ap.add_argument("--open", action="store_true", help="open neuroglancer in the default browser")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    httpd = serve(Path(a.dataset), a.port, a.host, a.codec, a.open, quiet=not a.verbose)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
