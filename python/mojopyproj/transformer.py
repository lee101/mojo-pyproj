from __future__ import annotations

import math
import shlex
from typing import Iterable

import numpy as np

from ._lib import addr, f64, lib
from .crs import CRS
from .enums import TransformDirection
from .exceptions import ProjError


def _direction(value) -> TransformDirection:
    if isinstance(value, TransformDirection):
        return value
    text = str(value).upper()
    if text.startswith("TRANSFORMDIRECTION."):
        text = text.rsplit(".", 1)[1]
    try:
        return TransformDirection[text]
    except KeyError as exc:
        raise ProjError(f"invalid transform direction: {value!r}") from exc


def _call_xy(
    name: str, x: np.ndarray, y: np.ndarray, *args, reuse_input: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    ox = x if reuse_input else np.empty_like(x)
    oy = y if reuse_input else np.empty_like(y)
    if x.size == 0:
        return ox, oy
    getattr(lib(), name)(addr(x), addr(y), addr(ox), addr(oy), x.size, *args)
    return ox, oy


def _geocentric(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, inverse: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ox = np.empty_like(x)
    oy = np.empty_like(y)
    oz = np.empty_like(z)
    if x.size == 0:
        return ox, oy, oz
    lib().mpj_geocentric(
        addr(x), addr(y), addr(z), addr(ox), addr(oy), addr(oz), x.size, int(inverse)
    )
    return ox, oy, oz


def _as_output(values: np.ndarray, scalar: bool, output_kind: str, shape):
    values = values.reshape(shape)
    if scalar:
        return float(values)
    if output_kind == "array":
        return values
    result = values.tolist()
    return tuple(result) if output_kind == "tuple" else result


class Transformer:
    """A pyproj-style reusable coordinate transformer backed by Mojo."""

    def __init__(
        self,
        source_crs: CRS | None,
        target_crs: CRS | None,
        *,
        always_xy: bool = False,
        helmert: dict[str, float] | None = None,
    ):
        self.source_crs = source_crs
        self.target_crs = target_crs
        self._always_xy = always_xy
        self._helmert = helmert

    @classmethod
    def from_crs(
        cls,
        crs_from,
        crs_to,
        always_xy: bool = False,
        area_of_interest=None,
        authority=None,
        accuracy=None,
        allow_ballpark=None,
        force_over: bool = False,
        only_best=None,
    ) -> "Transformer":
        unsupported = {
            "area_of_interest": area_of_interest,
            "authority": authority,
            "accuracy": accuracy,
            "allow_ballpark": allow_ballpark,
            "force_over": force_over or None,
            "only_best": only_best,
        }
        requested = [name for name, value in unsupported.items() if value is not None]
        if requested:
            raise ProjError(
                "unsupported Transformer.from_crs options: " + ", ".join(requested)
            )
        return cls(
            CRS.from_user_input(crs_from),
            CRS.from_user_input(crs_to),
            always_xy=always_xy,
        )

    @classmethod
    def from_pipeline(cls, proj_pipeline: str) -> "Transformer":
        tokens: dict[str, str] = {}
        words = shlex.split(proj_pipeline)
        normalized = [word.lstrip("+").lower() for word in words]
        if normalized.count("proj=helmert") != 1:
            raise ProjError("only +proj=helmert pipelines are covered")
        if "step" in normalized or sum(word.startswith("proj=") for word in normalized) != 1:
            raise ProjError("multi-step PROJ pipelines are not covered")
        allowed = {"proj", "x", "y", "z", "rx", "ry", "rz", "s", "convention"}
        for word in words:
            word = word.lstrip("+")
            if "=" in word:
                key, value = word.split("=", 1)
                if key.lower() not in allowed:
                    raise ProjError(f"unsupported Helmert parameter: {key}")
                tokens[key.lower()] = value
            elif word:
                raise ProjError(f"unsupported Helmert flag: {word}")
        convention = tokens.get("convention", "position_vector").lower()
        if convention not in {"position_vector", "coordinate_frame"}:
            raise ProjError("Helmert convention must be position_vector or coordinate_frame")
        try:
            params = {
                key: float(tokens.get(key, "0"))
                for key in ("x", "y", "z", "rx", "ry", "rz", "s")
            }
        except ValueError as exc:
            raise ProjError("Helmert parameters must be numeric") from exc
        if not all(math.isfinite(value) for value in params.values()):
            raise ProjError("Helmert parameters must be finite")
        if params["s"] == -1_000_000.0:
            raise ProjError("Helmert scale must not produce a zero scale factor")
        params["sign"] = 1.0 if convention == "position_vector" else -1.0
        return cls(None, None, helmert=params)

    @property
    def name(self) -> str:
        return "helmert" if self._helmert is not None else "pipeline"

    @property
    def description(self) -> str:
        if self._helmert is not None:
            return "Helmert transformation"
        assert self.source_crs is not None and self.target_crs is not None
        return f"axis order change + {self.source_crs.name} to {self.target_crs.name}"

    @property
    def accuracy(self) -> float:
        return 0.0

    @property
    def has_inverse(self) -> bool:
        return True

    def _crs_transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        reverse: bool,
        radians: bool,
        errcheck: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.source_crs is not None and self.target_crs is not None
        source = self.target_crs if reverse else self.source_crs
        target = self.source_crs if reverse else self.target_crs
        sd = source._definition
        td = target._definition

        if sd.kind == "geographic":
            lon, lat = (x, y) if self._always_xy else (y, x)
            if radians:
                lon = f64(np.rad2deg(lon))
                lat = f64(np.rad2deg(lat))
            invalid = np.abs(lat) > 90.0
            if errcheck and np.any(invalid):
                raise ProjError("transform error: invalid latitude")
        elif sd.kind == "web_mercator":
            lon, lat = _call_xy("mpj_web_mercator", x, y, 1)
            invalid = np.zeros(x.shape, dtype=bool)
        elif sd.kind == "world_mercator":
            lon, lat = _call_xy("mpj_world_mercator", x, y, 1)
            invalid = np.zeros(x.shape, dtype=bool)
        elif sd.kind == "utm":
            lon, lat = _call_xy(
                "mpj_utm", x, y, int(sd.zone), int(sd.south), 1
            )
            invalid = np.zeros(x.shape, dtype=bool)
        elif sd.kind == "geocentric":
            lon, lat, z = _geocentric(x, y, z, True)
            invalid = np.zeros(x.shape, dtype=bool)
        else:
            raise ProjError(f"unsupported source CRS kind: {sd.kind}")

        lon = f64(lon)
        lat = f64(lat)
        reuse_lon_lat = sd.kind != "geographic"
        if td.kind == "geographic":
            tx, ty = lon, lat
        elif td.kind == "web_mercator":
            tx, ty = _call_xy(
                "mpj_web_mercator", lon, lat, 0, reuse_input=reuse_lon_lat
            )
        elif td.kind == "world_mercator":
            tx, ty = _call_xy(
                "mpj_world_mercator", lon, lat, 0, reuse_input=reuse_lon_lat
            )
        elif td.kind == "utm":
            tx, ty = _call_xy(
                "mpj_utm",
                lon,
                lat,
                int(td.zone),
                int(td.south),
                0,
                reuse_input=reuse_lon_lat,
            )
        elif td.kind == "geocentric":
            tx, ty, z = _geocentric(lon, lat, z, False)
        else:
            raise ProjError(f"unsupported target CRS kind: {td.kind}")

        if td.kind == "geographic":
            if radians:
                tx = f64(np.deg2rad(tx))
                ty = f64(np.deg2rad(ty))
            if not self._always_xy:
                tx, ty = ty, tx
        if np.any(invalid):
            tx = tx.copy()
            ty = ty.copy()
            z = z.copy()
            tx[invalid] = math.inf
            ty[invalid] = math.inf
            z[invalid] = math.inf
        if errcheck and (np.any(~np.isfinite(tx)) or np.any(~np.isfinite(ty))):
            raise ProjError("transform error: non-finite coordinate result")
        return tx, ty, z

    def _helmert_transform(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, reverse: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._helmert is not None
        ox = np.empty_like(x)
        oy = np.empty_like(y)
        oz = np.empty_like(z)
        if x.size == 0:
            return ox, oy, oz
        p = self._helmert
        lib().mpj_helmert(
            addr(x), addr(y), addr(z), addr(ox), addr(oy), addr(oz), x.size,
            p["x"], p["y"], p["z"], p["rx"], p["ry"], p["rz"], p["s"], p["sign"],
            int(reverse),
        )
        return ox, oy, oz

    def transform(
        self,
        xx,
        yy,
        zz=None,
        tt=None,
        radians: bool = False,
        errcheck: bool = False,
        direction=TransformDirection.FORWARD,
        inplace: bool = False,
    ):
        del inplace
        supplied_z = zz is not None
        items = [np.asarray(xx), np.asarray(yy)]
        if supplied_z:
            items.append(np.asarray(zz))
        try:
            broadcast = np.broadcast_arrays(*items)
        except ValueError as exc:
            raise ProjError("x, y, and z must have broadcast-compatible shapes") from exc
        shape = broadcast[0].shape
        scalar = shape == ()
        if any(isinstance(value, np.ndarray) for value in (xx, yy, zz)):
            output_kind = "array"
        elif isinstance(xx, tuple):
            output_kind = "tuple"
        else:
            output_kind = "list"
        x = f64(broadcast[0].reshape(-1))
        y = f64(broadcast[1].reshape(-1))
        z = (
            f64(broadcast[2].reshape(-1))
            if supplied_z
            else np.zeros(x.size, dtype=np.float64)
        )
        # The Mojo kernels may update x/y buffers in place for composed
        # transforms. Keep each logical coordinate in distinct storage even
        # when NumPy broadcasting or repeated arguments produced aliases.
        if np.shares_memory(x, y):
            y = y.copy()
        if np.shares_memory(x, z) or np.shares_memory(y, z):
            z = z.copy()
        direct = _direction(direction)
        reverse = direct == TransformDirection.INVERSE
        if direct == TransformDirection.IDENT:
            ox, oy, oz = x.copy(), y.copy(), z.copy()
        elif self._helmert is not None:
            ox, oy, oz = self._helmert_transform(x, y, z, reverse)
        else:
            ox, oy, oz = self._crs_transform(x, y, z, reverse, radians, errcheck)
        result = [
            _as_output(ox, scalar, output_kind, shape),
            _as_output(oy, scalar, output_kind, shape),
        ]
        if supplied_z:
            result.append(_as_output(oz, scalar, output_kind, shape))
        if tt is not None:
            t = np.broadcast_to(np.asarray(tt), shape)
            result.append(_as_output(f64(t.reshape(-1)), scalar, output_kind, shape))
        return tuple(result)

    def itransform(
        self,
        points: Iterable,
        switch: bool = False,
        time_3rd: bool = False,
        radians: bool = False,
        errcheck: bool = False,
        direction=TransformDirection.FORWARD,
    ):
        rows = list(points)
        if not rows:
            return iter(())
        width = len(rows[0])
        if width not in (2, 3, 4) or any(len(row) != width for row in rows):
            raise ProjError("points must consistently contain 2, 3, or 4 values")
        data = np.asarray(rows, dtype=np.float64)
        if switch:
            data[:, [0, 1]] = data[:, [1, 0]]
        if width == 2:
            result = self.transform(
                data[:, 0], data[:, 1], radians=radians,
                errcheck=errcheck, direction=direction,
            )
        elif width == 3 and not time_3rd:
            result = self.transform(
                data[:, 0], data[:, 1], data[:, 2], radians=radians,
                errcheck=errcheck, direction=direction,
            )
        else:
            z = None if width == 3 else data[:, 2]
            t = data[:, 2] if width == 3 else data[:, 3]
            result = self.transform(
                data[:, 0], data[:, 1], z, t, radians=radians,
                errcheck=errcheck, direction=direction,
            )
        converted = np.column_stack(result)
        if switch:
            converted[:, [0, 1]] = converted[:, [1, 0]]
        return (tuple(row) for row in converted.tolist())
