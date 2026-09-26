"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-pyproj.so")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mpj_web_mercator": ([I, I, I, I, I, I], None),
    "mpj_world_mercator": ([I, I, I, I, I, I], None),
    "mpj_utm": ([I, I, I, I, I, I, I, I], None),
    "mpj_geocentric": ([I, I, I, I, I, I, I, I], None),
    "mpj_helmert": ([I, I, I, I, I, I, I] + [F] * 8 + [I], None),
    "mpj_utm_range": ([I] * 10, None),
    "mpj_geocentric_range": ([I] * 10, None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOPYPROJ_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    manifest = os.path.join(ROOT, "pixi.toml")
    if os.path.exists(pixi) and os.path.exists(manifest):
        return [pixi, "run", "--manifest-path", manifest, "mojo"]
    raise BuildError("mojo not found; run `pixi run build` or set MOJOPYPROJ_MOJO")


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + ["build", "--emit", "shared-lib", SRC, "-o", LIB]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:5000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def f64(values) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError("coordinate values must be real numbers, not complex")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"coordinate values must be real numbers, not {array.dtype}")
    if array.dtype.kind == "f" and array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("coordinate values wider than float64 are not supported")
    return np.ascontiguousarray(array, dtype=np.float64)


def addr(values: np.ndarray) -> int:
    if (
        values.dtype != np.float64
        or not values.flags.c_contiguous
        or not values.flags.aligned
    ):
        raise TypeError("FFI buffers must be aligned, C-contiguous float64 arrays")
    address = int(values.ctypes.data)
    if values.size and address == 0:
        raise ValueError("non-empty FFI buffers must have a non-null address")
    return address


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
