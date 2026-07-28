from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
from typing import Any

from .exceptions import CRSError


@dataclass(frozen=True)
class Axis:
    name: str
    abbrev: str
    direction: str
    unit_name: str


@dataclass(frozen=True)
class Ellipsoid:
    name: str = "WGS 84"
    semi_major_metre: float = 6378137.0
    semi_minor_metre: float = 6356752.314245179
    inverse_flattening: float = 298.257223563


@dataclass(frozen=True)
class _Definition:
    kind: str
    epsg: int | None = None
    zone: int | None = None
    south: bool = False
    title: str | None = None


_NAMES = {
    4326: "WGS 84",
    4979: "WGS 84",
    4978: "WGS 84",
    3857: "WGS 84 / Pseudo-Mercator",
    3395: "WGS 84 / World Mercator",
}


def _proj_tokens(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in shlex.split(text):
        token = token.lstrip("+")
        if "=" in token:
            key, value = token.split("=", 1)
            result[key.lower()] = value
        elif token:
            result[token.lower()] = "true"
    return result


def _from_epsg(code: int) -> _Definition:
    if code in (4326, 4979):
        return _Definition("geographic", code)
    if code == 4978:
        return _Definition("geocentric", code)
    if code == 3857:
        return _Definition("web_mercator", code)
    if code == 3395:
        return _Definition("world_mercator", code)
    if 32601 <= code <= 32660:
        return _Definition("utm", code, code - 32600, False)
    if 32701 <= code <= 32760:
        return _Definition("utm", code, code - 32700, True)
    raise CRSError(
        f"EPSG:{code} is outside mojo-pyproj's covered CRS subset"
    )


def _parse(value: Any) -> _Definition:
    if isinstance(value, CRS):
        return value._definition
    if hasattr(value, "crs") and isinstance(value.crs, CRS):
        return value.crs._definition
    if hasattr(value, "to_epsg") and callable(value.to_epsg):
        code = value.to_epsg()
        if code is not None:
            return _from_epsg(int(code))
    if isinstance(value, int):
        return _from_epsg(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if str(value[0]).upper() == "EPSG":
            return _from_epsg(int(value[1]))
    if isinstance(value, dict):
        params = {str(k).lower(): str(v) for k, v in value.items()}
        if "init" in params:
            return _parse(params["init"])
        return _from_params(params)
    if not isinstance(value, str):
        raise CRSError(f"invalid CRS input: {value!r}")
    text = value.strip()
    upper = text.upper()
    if upper in {"OGC:CRS84", "CRS84"}:
        return _Definition("geographic", 4326)
    match = re.search(r"(?:EPSG(?::|::)|EPSG/0/)(\d+)$", upper)
    if match:
        return _from_epsg(int(match.group(1)))
    if text.isdigit():
        return _from_epsg(int(text))
    if "+proj" in text.lower() or text.lower().startswith("proj="):
        return _from_params(_proj_tokens(text))
    raise CRSError(f"unsupported CRS definition: {value!r}")


def _from_params(params: dict[str, str]) -> _Definition:
    proj = params.get("proj", params.get("projection", "")).lower()
    common = {"proj", "projection", "datum", "ellps", "units", "type", "no_defs"}
    allowed = {
        "longlat": common,
        "latlong": common,
        "lonlat": common,
        "geocent": common,
        "utm": common | {"zone", "south"},
        "merc": common | {"a", "b"},
    }.get(proj, common)
    unsupported = sorted(set(params) - allowed)
    if unsupported:
        raise CRSError(
            f"unsupported +proj={proj or '<missing>'} parameters: "
            + ", ".join(unsupported)
        )
    datum = params.get("datum", "WGS84").upper()
    ellps = params.get("ellps", "WGS84").upper()
    if datum not in {"WGS84", "WGS_1984"} or ellps not in {"WGS84", "WGS_1984"}:
        raise CRSError("only the WGS84 ellipsoid is covered")
    if proj in {"longlat", "latlong", "lonlat"}:
        return _Definition("geographic", 4326)
    if proj == "geocent":
        return _Definition("geocentric", 4978)
    if proj == "utm":
        try:
            zone = int(params["zone"])
        except (KeyError, ValueError) as exc:
            raise CRSError("UTM definitions require +zone=1..60") from exc
        if not 1 <= zone <= 60:
            raise CRSError("UTM zone must be between 1 and 60")
        south = "south" in params and params["south"].lower() not in {"false", "0"}
        return _Definition("utm", 32700 + zone if south else 32600 + zone, zone, south)
    if proj == "merc":
        try:
            a = float(params.get("a", "6378137"))
            b = float(params.get("b", "6356752.314245179"))
        except ValueError as exc:
            raise CRSError("Mercator +a and +b must be numeric") from exc
        if abs(a - 6378137.0) > 1e-6:
            raise CRSError("only the WGS84 semi-major axis is covered")
        if abs(b - a) < 1e-6:
            return _Definition("web_mercator", 3857)
        if abs(b - 6356752.314245179) > 1e-6:
            raise CRSError("only spherical or WGS84 Mercator is covered")
        return _Definition("world_mercator", 3395)
    raise CRSError(f"unsupported +proj={proj or '<missing>'}")


class CRS:
    """A compact pyproj-compatible CRS object for the covered WGS84 subset."""

    def __init__(self, projparams=None, **kwargs):
        if kwargs:
            params = dict(kwargs)
            if projparams is not None:
                if isinstance(projparams, dict):
                    params = {**projparams, **params}
                else:
                    raise CRSError("keyword parameters require a mapping CRS")
            projparams = params
        self._definition = _parse(projparams)

    @classmethod
    def from_user_input(cls, value, **kwargs) -> "CRS":
        return cls(value, **kwargs)

    @classmethod
    def from_epsg(cls, code: int) -> "CRS":
        return cls(int(code))

    @classmethod
    def from_string(cls, value: str) -> "CRS":
        return cls(value)

    @classmethod
    def from_dict(cls, value: dict) -> "CRS":
        return cls(value)

    @property
    def name(self) -> str:
        d = self._definition
        if d.kind == "utm":
            return f"WGS 84 / UTM zone {d.zone}{'S' if d.south else 'N'}"
        return _NAMES.get(d.epsg, d.title or "unknown")

    @property
    def type_name(self) -> str:
        if self.is_geographic:
            return "Geographic 2D CRS" if self.to_epsg() == 4326 else "Geographic 3D CRS"
        if self.is_geocentric:
            return "Geocentric CRS"
        return "Projected CRS"

    @property
    def is_geographic(self) -> bool:
        return self._definition.kind == "geographic"

    @property
    def is_projected(self) -> bool:
        return self._definition.kind in {"web_mercator", "world_mercator", "utm"}

    @property
    def is_geocentric(self) -> bool:
        return self._definition.kind == "geocentric"

    @property
    def is_bound(self) -> bool:
        return False

    @property
    def ellipsoid(self) -> Ellipsoid:
        return Ellipsoid()

    @property
    def axis_info(self) -> list[Axis]:
        if self.is_geographic:
            axes = [
                Axis("Geodetic latitude", "Lat", "north", "degree"),
                Axis("Geodetic longitude", "Lon", "east", "degree"),
            ]
            if self.to_epsg() == 4979:
                axes.append(Axis("Ellipsoidal height", "h", "up", "metre"))
            return axes
        if self.is_geocentric:
            return [
                Axis("Geocentric X", "X", "geocentricX", "metre"),
                Axis("Geocentric Y", "Y", "geocentricY", "metre"),
                Axis("Geocentric Z", "Z", "geocentricZ", "metre"),
            ]
        return [
            Axis("Easting", "E", "east", "metre"),
            Axis("Northing", "N", "north", "metre"),
        ]

    def to_epsg(self, min_confidence: int = 70) -> int | None:
        del min_confidence
        return self._definition.epsg

    def to_authority(self, auth_name=None, min_confidence: int = 70):
        del min_confidence
        if auth_name is not None and str(auth_name).upper() != "EPSG":
            return None
        code = self.to_epsg()
        return ("EPSG", str(code)) if code is not None else None

    def to_string(self, auth_name=None, warnings=False) -> str:
        del auth_name, warnings
        code = self.to_epsg()
        return f"EPSG:{code}" if code is not None else self.to_proj4()

    def to_proj4(self, version=None) -> str:
        del version
        d = self._definition
        if d.kind == "geographic":
            return "+proj=longlat +datum=WGS84 +no_defs +type=crs"
        if d.kind == "geocentric":
            return "+proj=geocent +datum=WGS84 +units=m +no_defs +type=crs"
        if d.kind == "utm":
            south = " +south" if d.south else ""
            return f"+proj=utm +zone={d.zone}{south} +datum=WGS84 +units=m +no_defs +type=crs"
        if d.kind == "web_mercator":
            return "+proj=merc +a=6378137 +b=6378137 +units=m +no_defs +type=crs"
        return "+proj=merc +datum=WGS84 +units=m +no_defs +type=crs"

    def to_dict(self) -> dict[str, Any]:
        d = self._definition
        if d.kind == "utm":
            result: dict[str, Any] = {
                "proj": "utm", "zone": d.zone, "datum": "WGS84", "units": "m"
            }
            if d.south:
                result["south"] = True
            return result
        if d.kind == "geographic":
            return {"proj": "longlat", "datum": "WGS84"}
        if d.kind == "geocentric":
            return {"proj": "geocent", "datum": "WGS84", "units": "m"}
        if d.kind == "web_mercator":
            return {"proj": "merc", "a": 6378137, "b": 6378137, "units": "m"}
        return {"proj": "merc", "datum": "WGS84", "units": "m"}

    def equals(self, other, ignore_axis_order: bool = False) -> bool:
        del ignore_axis_order
        try:
            return self._definition == CRS.from_user_input(other)._definition
        except CRSError:
            return False

    def is_exact_same(self, other) -> bool:
        return self.equals(other)

    def __eq__(self, other) -> bool:
        return self.equals(other)

    def __hash__(self) -> int:
        return hash(self._definition)

    def __repr__(self) -> str:
        return f"<CRS: {self.to_string()}>\nName: {self.name}\nType: {self.type_name}"

    def __str__(self) -> str:
        return self.to_string()
