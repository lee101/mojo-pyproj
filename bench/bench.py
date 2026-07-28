"""Locked benchmark against pyproj/PROJ on identical NumPy inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pyproj

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojopyproj as mp  # noqa: E402


def best_time(fn, repeat: int = 5) -> float:
    fn()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


N = 1_000_000
rng = np.random.default_rng(2026)
lon = np.ascontiguousarray(rng.uniform(-78.0, -72.0, N))
lat = np.ascontiguousarray(rng.uniform(35.0, 45.0, N))
height = np.ascontiguousarray(rng.uniform(-100.0, 3000.0, N))


def transformer_case(label, source, target, *coordinates):
    ours = mp.Transformer.from_crs(source, target, always_xy=True)
    reference = pyproj.Transformer.from_crs(source, target, always_xy=True)
    return label, lambda: ours.transform(*coordinates), lambda: reference.transform(*coordinates)


web_xy = mp.Transformer.from_crs(4326, 3857, always_xy=True).transform(lon, lat)
utm_xy = mp.Transformer.from_crs(4326, 32618, always_xy=True).transform(lon, lat)
ecef_xyz = mp.Transformer.from_crs(4979, 4978, always_xy=True).transform(
    lon, lat, height
)

helmert_pipeline = (
    "+proj=helmert +x=1 +y=2 +z=3 +rx=0.1 +ry=-0.2 +rz=0.3 "
    "+s=4 +convention=position_vector"
)
helmert_ours = mp.Transformer.from_pipeline(helmert_pipeline)
helmert_reference = pyproj.Transformer.from_pipeline(helmert_pipeline)

CASES = [
    transformer_case("Web Mercator forward", 4326, 3857, lon, lat),
    transformer_case("World Mercator forward", 4326, 3395, lon, lat),
    transformer_case("UTM zone 18N forward", 4326, 32618, lon, lat),
    transformer_case("UTM zone 18N inverse", 32618, 4326, *utm_xy),
    transformer_case("WGS84 geographic to ECEF", 4979, 4978, lon, lat, height),
    transformer_case("WGS84 ECEF to geographic", 4978, 4979, *ecef_xyz),
    transformer_case("UTM 18N to Web Mercator", 32618, 3857, *utm_xy),
    (
        "7-parameter Helmert",
        lambda: helmert_ours.transform(*ecef_xyz),
        lambda: helmert_reference.transform(*ecef_xyz),
    ),
]


def main() -> None:
    print(f"Machine: {cpu_name()}; {os.cpu_count()} logical CPUs; {platform.system()} {platform.machine()}")
    print(f"Runtime: Python {platform.python_version()}; pyproj {pyproj.__version__}; PROJ {pyproj.proj_version_str}")
    print(f"Dataset: {N:,} float64 coordinate tuples; best of 5 warm runs")
    print()
    print("| Operation | mojo-pyproj | pyproj | Speedup | Result |")
    print("|---|---:|---:|---:|:---|")
    for label, ours, reference in CASES:
        mojo_seconds = best_time(ours)
        pyproj_seconds = best_time(reference)
        ratio = pyproj_seconds / mojo_seconds
        result = "faster" if ratio >= 1.0 else "slower"
        print(
            f"| {label} | {mojo_seconds * 1000:.2f} ms | "
            f"{pyproj_seconds * 1000:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
