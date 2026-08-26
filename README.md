# mojo-pyproj

`mojo-pyproj` is a standalone Mojo implementation of the compute-heavy core of a
useful `pyproj` subset. It provides familiar `CRS`, `Proj`, `Transformer`,
`transform`, and `itransform` APIs while running bulk coordinate operations in a
compiled Mojo shared library.

This is a focused port, not a wrapper around PROJ. The upstream `pyproj` package
is installed only as a development dependency for numerical parity tests and
benchmarks.

## Covered subset

The following WGS84 coordinate reference systems and operations are implemented:

- EPSG:4326 and EPSG:4979 geographic coordinates
- EPSG:3857 WGS 84 / Pseudo-Mercator
- EPSG:3395 WGS 84 / World Mercator
- EPSG:32601 through EPSG:32660, northern WGS84 UTM zones
- EPSG:32701 through EPSG:32760, southern WGS84 UTM zones
- EPSG:4978 WGS84 geocentric coordinates
- Direct and composed transforms between every CRS above
- Three- and seven-parameter `+proj=helmert` pipelines, with both
  `position_vector` and `coordinate_frame` conventions
- Scalar, list, tuple, and NumPy array inputs; 2D, 3D, and time pass-through;
  `always_xy`, radians, inverse direction, and `errcheck`
- Common CRS construction forms including EPSG integers and strings, authority
  tuples, dictionaries, and covered PROJ strings

The tests compare every numerical kernel with `pyproj 3.7.2` / PROJ 9.8.1.
Across the randomized test domains, forward UTM differs by less than 0.2 mm,
Mercator by less than 0.05 mm, and WGS84 geographic-to-ECEF by less than
0.00000002 m.

Not covered are arbitrary EPSG definitions, non-WGS84 ellipsoids, grid-shift
files, vertical geoids, time-dependent reference frames, WKT parsing, the PROJ
database or network, arbitrary PROJ pipelines, and projections such as Lambert
Conformal Conic or Albers Equal Area. Use upstream `pyproj` when an operation
falls outside the explicitly covered subset; unsupported definitions raise
`CRSError` instead of silently choosing an approximation.

## Install and run

The repository pins the tested Mojo nightly:

```bash
pixi install
pixi run build
pixi run test
```

The Pixi activation adds `python/` to `PYTHONPATH`. A source-tree example:

```python
from mojopyproj import Transformer

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
easting, northing = to_utm.transform(-75.0, 40.0)
print(round(easting, 3), round(northing, 3))
# 500000.0 4427757.219

lon, lat = to_utm.transform(easting, northing, direction="INVERSE")
print(round(lon, 8), round(lat, 8))
# -75.0 40.0
```

`mojo_pyproj` is also provided as an import alias. `mojopyproj` is the canonical
module name used in this repository so it can coexist with upstream `pyproj`
during parity testing.

## Benchmark

Run benchmarks only through the locked Pixi task:

```bash
pixi run bench
```

These are real best-of-five warm measurements from the current repository, over
one million float64 coordinate tuples.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 logical CPUs, Linux
x86_64. Runtime: Python 3.13.14, pyproj 3.7.2, PROJ 9.8.1.

| Operation | mojo-pyproj | pyproj | Speedup | Result |
|---|---:|---:|---:|:---|
| Web Mercator forward | 37.48 ms | 139.97 ms | 3.73x | faster |
| World Mercator forward | 48.48 ms | 152.52 ms | 3.15x | faster |
| UTM zone 18N forward | 27.61 ms | 252.31 ms | 9.14x | faster |
| UTM zone 18N inverse | 87.89 ms | 276.89 ms | 3.15x | faster |
| WGS84 geographic to ECEF | 18.40 ms | 141.39 ms | 7.68x | faster |
| WGS84 ECEF to geographic | 31.14 ms | 159.17 ms | 5.11x | faster |
| UTM 18N to Web Mercator | 103.28 ms | 360.31 ms | 3.49x | faster |
| 7-parameter Helmert | 10.29 ms | 51.24 ms | 4.98x | faster |

The UTM and geocentric kernels split large independent coordinate arrays across
the machine's physical cores. Mercator and Helmert remain serial because thread
launch and contention costs outweighed their per-point work. Small calls still
pay roughly the normal ctypes call cost, so these results should not be read as
a claim that every scalar call is faster than PROJ.

No GPU path is shipped. The inverse geocentric and UTM kernels have enough
arithmetic intensity to be candidates, but the pinned Mojo toolchain cannot
compile the required float64 trigonometric operations for NVIDIA GPUs. Using
float32 would violate the existing pyproj parity tolerances, so CPU execution is
kept rather than adding a GPU path that changes results.

## How it works

All numerical code lives in one Mojo compilation unit,
`src/kernels.mojo`, and `build/build.sh` emits
`dist/libmojo-pyproj.so`. The Python layer resolves CRS paths and preserves
pyproj-style argument and result containers. It then supplies contiguous
row-major NumPy `float64` buffers to the shared library through `ctypes`.

Buffers cross the C ABI as integer addresses. Mojo reconstructs
`Pointer[Float64, AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions, writes into caller-owned output buffers, and performs no
FFI-side allocation. Projection operations with at least 65,536 coordinates are
divided across physical cores for the arithmetic-heavy UTM and geocentric
kernels; smaller arrays stay on one worker to avoid scheduling overhead. The
geocentric inverse and Helmert kernels use SIMD with scalar tails. Composed
projected-to-projected operations reuse caller-owned intermediate longitude and
latitude buffers in place.

The UTM implementation uses sixth-order transverse Mercator terms and a
sixth-order third-flattening meridional arc, followed by Newton refinement for
the inverse footpoint latitude. Geocentric conversion uses the WGS84 ellipsoid
and an iterated Bowring inverse. Helmert inverse behavior deliberately matches
PROJ's linearized rotation convention rather than replacing it with a different
exact-matrix operation.
