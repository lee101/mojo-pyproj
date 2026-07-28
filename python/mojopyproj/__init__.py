"""A useful, numerically tested subset of pyproj accelerated by Mojo."""

from __future__ import annotations

from .crs import CRS
from .enums import ProjVersion, TransformDirection
from .exceptions import CRSError, ProjError
from .proj import Proj
from .transformer import Transformer

__version__ = "0.1.0"


def transform(
    p1,
    p2,
    x,
    y,
    z=None,
    radians: bool = False,
    errcheck: bool = False,
    always_xy: bool = False,
):
    source = p1.crs if isinstance(p1, Proj) else p1
    target = p2.crs if isinstance(p2, Proj) else p2
    return Transformer.from_crs(source, target, always_xy=always_xy).transform(
        x, y, z, radians=radians, errcheck=errcheck
    )


def itransform(
    p1,
    p2,
    points,
    radians: bool = False,
    errcheck: bool = False,
    always_xy: bool = False,
):
    source = p1.crs if isinstance(p1, Proj) else p1
    target = p2.crs if isinstance(p2, Proj) else p2
    return Transformer.from_crs(source, target, always_xy=always_xy).itransform(
        points, radians=radians, errcheck=errcheck
    )


__all__ = [
    "CRS",
    "CRSError",
    "Proj",
    "ProjError",
    "ProjVersion",
    "TransformDirection",
    "Transformer",
    "itransform",
    "transform",
]
