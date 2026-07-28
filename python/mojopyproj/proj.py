from __future__ import annotations

from .crs import CRS
from .transformer import Transformer


class Proj:
    """Projection callable matching pyproj.Proj for the covered CRS subset."""

    def __init__(self, projparams=None, preserve_units: bool = True, **kwargs):
        del preserve_units
        self.crs = CRS(projparams, **kwargs)
        self._transformer = Transformer.from_crs(4326, self.crs, always_xy=True)
        self.srs = self.crs.to_proj4()

    @property
    def definition(self) -> str:
        return self.srs

    def is_latlong(self) -> bool:
        return self.crs.is_geographic

    def is_geocent(self) -> bool:
        return self.crs.is_geocentric

    def __call__(
        self,
        longitude,
        latitude,
        inverse: bool = False,
        errcheck: bool = False,
        radians: bool = False,
    ):
        direction = "INVERSE" if inverse else "FORWARD"
        return self._transformer.transform(
            longitude,
            latitude,
            radians=radians,
            errcheck=errcheck,
            direction=direction,
        )

    def __repr__(self) -> str:
        return f"<Other Coordinate Operation Transformer: {self.crs.name}>"
