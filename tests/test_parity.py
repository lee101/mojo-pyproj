import math

import numpy as np
import pytest
import pyproj

import mojopyproj as mp


def pair(src, dst, always_xy=True):
    return (
        mp.Transformer.from_crs(src, dst, always_xy=always_xy),
        pyproj.Transformer.from_crs(src, dst, always_xy=always_xy),
    )


@pytest.mark.parametrize(
    "dst,xy,tolerance",
    [
        (3857, (-75.0, 40.0), 3e-5),
        (3395, (2.2945, 48.8584), 3e-5),
    ],
)
def test_mercator_scalar_parity(dst, xy, tolerance):
    ours, reference = pair(4326, dst)
    assert ours.transform(*xy) == pytest.approx(reference.transform(*xy), abs=tolerance)


def test_epsg_axis_order_parity():
    ours, reference = pair(4326, 3857, always_xy=False)
    assert ours.transform(40.0, -75.0) == pytest.approx(
        reference.transform(40.0, -75.0), abs=3e-5
    )
    x, y = ours.transform(40.0, -75.0)
    assert ours.transform(x, y, direction="INVERSE") == pytest.approx(
        (40.0, -75.0), abs=2e-9
    )


@pytest.mark.parametrize("dst", [3857, 3395])
def test_mercator_random_array_parity(dst):
    rng = np.random.default_rng(dst)
    lon = rng.uniform(-179.0, 179.0, 10_000)
    lat = rng.uniform(-80.0, 80.0, 10_000)
    ours, reference = pair(4326, dst)
    actual = ours.transform(lon, lat)
    expected = reference.transform(lon, lat)
    assert np.allclose(actual[0], expected[0], atol=5e-5, rtol=0)
    assert np.allclose(actual[1], expected[1], atol=5e-5, rtol=0)


@pytest.mark.parametrize(
    "epsg,lon,lat",
    [
        (32601, -177.0, 60.0),
        (32618, -75.0, 40.0),
        (32631, 3.0, 0.0),
        (32632, 12.0, 84.0),
        (32756, 151.0, -33.0),
    ],
)
def test_utm_published_locations(epsg, lon, lat):
    ours, reference = pair(4326, epsg)
    assert ours.transform(lon, lat) == pytest.approx(
        reference.transform(lon, lat), abs=2e-4
    )


@pytest.mark.parametrize(
    "epsg,lon0,low,high",
    [
        (32618, -75.0, 0.0, 83.5),
        (32632, 9.0, 0.0, 83.5),
        (32756, 153.0, -79.5, -0.1),
    ],
)
def test_utm_random_forward_and_inverse(epsg, lon0, low, high):
    rng = np.random.default_rng(epsg)
    lon = lon0 + rng.uniform(-2.9, 2.9, 2_000)
    lat = rng.uniform(low, high, 2_000)
    ours, reference = pair(4326, epsg)
    projected = ours.transform(lon, lat)
    expected = reference.transform(lon, lat)
    assert np.allclose(projected[0], expected[0], atol=2e-4, rtol=0)
    assert np.allclose(projected[1], expected[1], atol=2e-4, rtol=0)
    back = ours.transform(*projected, direction=mp.TransformDirection.INVERSE)
    ref_back = reference.transform(*projected, direction="INVERSE")
    assert np.allclose(back[0], ref_back[0], atol=1.1e-8, rtol=0)
    assert np.allclose(back[1], ref_back[1], atol=1.1e-8, rtol=0)


@pytest.mark.parametrize(
    "epsg",
    [*range(32601, 32661), *range(32701, 32761)],
)
def test_every_documented_utm_zone_forward_and_inverse(epsg):
    south = epsg >= 32701
    zone = epsg - (32700 if south else 32600)
    lon = zone * 6.0 - 183.0
    lat = -40.0 if south else 40.0
    ours, reference = pair(4326, epsg)
    actual = ours.transform(lon, lat)
    expected = reference.transform(lon, lat)
    assert actual == pytest.approx(expected, abs=2e-4)
    assert ours.transform(*actual, direction="INVERSE") == pytest.approx(
        (lon, lat), abs=1.1e-8
    )


@pytest.mark.parametrize(
    "lon,lat,height",
    [
        (-75.0, 40.0, 100.0),
        (0.0, 0.0, 0.0),
        (151.2, -33.8, 1250.0),
        (25.0, 80.0, -50.0),
    ],
)
def test_geocentric_scalar_parity(lon, lat, height):
    ours, reference = pair(4979, 4978)
    xyz = ours.transform(lon, lat, height)
    assert xyz == pytest.approx(reference.transform(lon, lat, height), abs=2e-8)
    llh = ours.transform(*xyz, direction="INVERSE")
    assert llh == pytest.approx((lon, lat, height), abs=2e-6)


def test_geocentric_random_array_parity():
    rng = np.random.default_rng(8)
    lon = rng.uniform(-180, 180, 5_000)
    lat = rng.uniform(-88, 88, 5_000)
    height = rng.uniform(-500, 20_000, 5_000)
    ours, reference = pair(4979, 4978)
    actual = ours.transform(lon, lat, height)
    expected = reference.transform(lon, lat, height)
    for got, want in zip(actual, expected):
        assert np.allclose(got, want, atol=2e-8, rtol=0)
    back = ours.transform(*actual, direction="INVERSE")
    assert np.allclose(back[0], lon, atol=2e-12, rtol=0)
    assert np.allclose(back[1], lat, atol=4e-11, rtol=0)
    assert np.allclose(back[2], height, atol=5e-6, rtol=0)


@pytest.mark.parametrize("size", [3, 5, 9, 17])
def test_geocentric_inverse_simd_tail(size):
    lon = np.linspace(-170.0, 170.0, size)
    lat = np.linspace(-80.0, 80.0, size)
    height = np.linspace(-100.0, 10_000.0, size)
    ours, reference = pair(4979, 4978)
    xyz = reference.transform(lon, lat, height)
    actual = ours.transform(*xyz, direction="INVERSE")
    expected = reference.transform(*xyz, direction="INVERSE")
    assert np.allclose(actual[0], expected[0], atol=2e-12, rtol=0)
    assert np.allclose(actual[1], expected[1], atol=4e-11, rtol=0)
    assert np.allclose(actual[2], expected[2], atol=5e-6, rtol=0)


@pytest.mark.parametrize("size", [65_535, 65_536])
def test_projection_parallel_threshold(size):
    lon = np.linspace(-77.0, -73.0, size)
    lat = np.linspace(36.0, 44.0, size)
    ours, reference = pair(4326, 32618)
    actual = ours.transform(lon, lat)
    expected = reference.transform(lon, lat)
    assert np.allclose(actual, expected, atol=2e-4, rtol=0)


def test_projection_to_projection_parity():
    lon = np.linspace(-77.5, -72.5, 1_000)
    lat = np.linspace(37.0, 44.0, 1_000)
    utm = mp.Transformer.from_crs(4326, 32618, always_xy=True).transform(lon, lat)
    original = tuple(values.copy() for values in utm)
    ours, reference = pair(32618, 3857)
    actual = ours.transform(*utm)
    expected = reference.transform(*utm)
    assert np.allclose(actual[0], expected[0], atol=2e-3, rtol=0)
    assert np.allclose(actual[1], expected[1], atol=2e-3, rtol=0)
    assert np.array_equal(utm[0], original[0])
    assert np.array_equal(utm[1], original[1])


def test_height_is_preserved_by_projected_crs():
    ours, reference = pair(4979, 32618)
    actual = ours.transform(-75.0, 40.0, 123.5)
    expected = reference.transform(-75.0, 40.0, 123.5)
    assert actual == pytest.approx(expected, abs=2e-4)


def test_radians_parity():
    ours, reference = pair(4326, 3857)
    lon, lat = math.radians(-75), math.radians(40)
    assert ours.transform(lon, lat, radians=True) == pytest.approx(
        reference.transform(lon, lat, radians=True), abs=3e-5
    )
    x, y = ours.transform(lon, lat, radians=True)
    assert ours.transform(x, y, radians=True, direction="INVERSE") == pytest.approx(
        (lon, lat), abs=3e-12
    )


@pytest.mark.parametrize(
    "make,kind",
    [
        (lambda: (-75.0, 40.0), float),
        (lambda: ([-75.0], [40.0]), list),
        (lambda: ((-75.0,), (40.0,)), tuple),
        (lambda: (np.array([-75.0]), np.array([40.0])), np.ndarray),
    ],
)
def test_input_container_parity(make, kind):
    x, y = make()
    ours, reference = pair(4326, 3857)
    actual = ours.transform(x, y)
    expected = reference.transform(x, y)
    assert isinstance(actual[0], kind)
    assert isinstance(actual[1], kind)
    assert np.allclose(actual, expected, atol=3e-5)


def test_broadcast_inputs_are_supported():
    transformer = mp.Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = transformer.transform(np.array([-75.0, -74.0]), 40.0)
    assert x.shape == (2,) and y.shape == (2,)


def test_empty_and_strided_arrays_are_safe():
    transformer = mp.Transformer.from_crs(4326, 3857, always_xy=True)
    x, y = transformer.transform(np.array([], dtype=np.float64), np.array([]))
    assert x.shape == (0,) and y.shape == (0,)
    source = np.arange(20, dtype=np.float32)
    x, y = transformer.transform(source[::2], np.full(10, 40, dtype=np.int16))
    expected = pyproj.Transformer.from_crs(4326, 3857, always_xy=True).transform(
        source[::2], np.full(10, 40, dtype=np.int16)
    )
    assert np.allclose((x, y), expected, atol=3e-5, rtol=0)


def test_ffi_dtype_contract_rejects_lossy_or_non_numeric_inputs():
    transformer = mp.Transformer.from_crs(4326, 3857, always_xy=True)
    with pytest.raises(TypeError, match="complex"):
        transformer.transform(np.array([1 + 2j]), np.array([40.0]))
    with pytest.raises(TypeError, match="real numbers"):
        transformer.transform(np.array(["-75"]), np.array(["40"]))
    if np.dtype(np.longdouble).itemsize > np.dtype(np.float64).itemsize:
        with pytest.raises(TypeError, match="wider than float64"):
            transformer.transform(
                np.array([-75], dtype=np.longdouble),
                np.array([40], dtype=np.longdouble),
            )


def test_repeated_input_buffer_does_not_alias_composed_outputs():
    values = np.linspace(400_000.0, 500_000.0, 100)
    ours, reference = pair(32618, 3857)
    actual = ours.transform(values, values)
    expected = reference.transform(values, values)
    assert np.allclose(actual, expected, atol=2e-3, rtol=0)


def test_invalid_latitude_and_errcheck():
    transformer = mp.Transformer.from_crs(4326, 3857, always_xy=True)
    assert transformer.transform(0.0, 91.0) == (math.inf, math.inf)
    with pytest.raises(mp.ProjError, match="invalid latitude"):
        transformer.transform(0.0, 91.0, errcheck=True)
    with pytest.raises(mp.ProjError, match="non-finite"):
        transformer.transform(0.0, 90.0, errcheck=True)


@pytest.mark.parametrize("convention", ["position_vector", "coordinate_frame"])
def test_seven_parameter_helmert_parity(convention):
    pipeline = (
        "+proj=helmert +x=1 +y=2 +z=3 +rx=0.1 +ry=-0.2 +rz=0.3 "
        f"+s=4 +convention={convention}"
    )
    ours = mp.Transformer.from_pipeline(pipeline)
    reference = pyproj.Transformer.from_pipeline(pipeline)
    xyz = (3657660.66, 255768.55, 5201382.11)
    transformed = ours.transform(*xyz)
    assert transformed == pytest.approx(reference.transform(*xyz), abs=2e-9)
    assert ours.transform(*transformed, direction="INVERSE") == pytest.approx(
        reference.transform(*transformed, direction="INVERSE"), abs=2e-9
    )


def test_helmert_array_translation():
    pipeline = "+proj=helmert +x=10 +y=-5 +z=2"
    values = np.arange(3000, dtype=np.float64)
    ours = mp.Transformer.from_pipeline(pipeline).transform(values, values, values)
    reference = pyproj.Transformer.from_pipeline(pipeline).transform(
        values, values, values
    )
    assert np.array_equal(ours, reference)


def test_unsupported_pipeline_and_transformer_options_raise():
    with pytest.raises(mp.ProjError, match="covered|multi-step"):
        mp.Transformer.from_pipeline(
            "+proj=pipeline +step +proj=helmert +x=1 +step +proj=helmert +y=2"
        )
    with pytest.raises(mp.ProjError, match="unsupported Helmert parameter"):
        mp.Transformer.from_pipeline("+proj=helmert +dx=1")
    with pytest.raises(mp.ProjError, match="unsupported Transformer"):
        mp.Transformer.from_crs(4326, 3857, force_over=True)


@pytest.mark.parametrize("size", [3, 5, 9, 17])
def test_helmert_simd_tail_parity(size):
    pipeline = (
        "+proj=helmert +x=1 +y=2 +z=3 +rx=0.1 +ry=-0.2 +rz=0.3 "
        "+s=4 +convention=position_vector"
    )
    ours = mp.Transformer.from_pipeline(pipeline)
    reference = pyproj.Transformer.from_pipeline(pipeline)
    x = np.linspace(3_000_000.0, 4_000_000.0, size)
    y = np.linspace(-1_000_000.0, 1_000_000.0, size)
    z = np.linspace(4_500_000.0, 5_500_000.0, size)
    actual = ours.transform(x, y, z)
    expected = reference.transform(x, y, z)
    assert np.allclose(actual, expected, atol=2e-9, rtol=0)
    actual_inverse = ours.transform(*actual, direction="INVERSE")
    expected_inverse = reference.transform(*actual, direction="INVERSE")
    assert np.allclose(actual_inverse, expected_inverse, atol=2e-9, rtol=0)


@pytest.mark.parametrize("size", [262_143, 262_144])
def test_helmert_parallel_threshold(size):
    transformer = mp.Transformer.from_pipeline("+proj=helmert +x=10 +y=-5 +z=2")
    values = np.arange(size, dtype=np.float64)
    x, y, z = transformer.transform(values, values, values)
    assert np.array_equal(x, values + 10.0)
    assert np.array_equal(y, values - 5.0)
    assert np.array_equal(z, values + 2.0)


@pytest.mark.parametrize(
    "value,epsg",
    [
        (4326, 4326),
        ("4326", 4326),
        ("EPSG:3857", 3857),
        ("urn:ogc:def:crs:EPSG::4978", 4978),
        (("EPSG", 32618), 32618),
        ({"proj": "utm", "zone": 56, "south": True}, 32756),
        ("+proj=utm +zone=32 +datum=WGS84", 32632),
        ("+proj=merc +a=6378137 +b=6378137", 3857),
    ],
)
def test_crs_input_forms(value, epsg):
    ours = mp.CRS.from_user_input(value)
    reference = pyproj.CRS.from_user_input(value)
    assert ours.to_epsg() == epsg
    if reference.to_epsg() is not None:
        assert ours.to_epsg() == reference.to_epsg()
    assert ours.to_authority() == ("EPSG", str(epsg))


def test_crs_properties():
    geographic = mp.CRS.from_epsg(4326)
    projected = mp.CRS.from_epsg(32618)
    geocentric = mp.CRS.from_epsg(4978)
    assert geographic.is_geographic and not geographic.is_projected
    assert projected.is_projected and projected.name == "WGS 84 / UTM zone 18N"
    assert geocentric.is_geocentric
    assert geographic.ellipsoid.semi_major_metre == 6378137.0
    assert [axis.direction for axis in geographic.axis_info] == ["north", "east"]
    assert mp.CRS.from_dict(projected.to_dict()) == projected


@pytest.mark.parametrize(
    "epsg", [4326, 4979, 4978, 3857, 3395, *range(32601, 32661), *range(32701, 32761)]
)
def test_every_documented_epsg_is_constructible(epsg):
    assert mp.CRS.from_epsg(epsg).to_epsg() == epsg


def test_crs_accepts_upstream_crs():
    assert mp.CRS.from_user_input(pyproj.CRS.from_epsg(3395)).to_epsg() == 3395


def test_unsupported_crs_is_explicit():
    with pytest.raises(mp.CRSError, match="outside"):
        mp.CRS.from_epsg(27700)
    with pytest.raises(mp.CRSError, match="only the WGS84"):
        mp.CRS("+proj=utm +zone=18 +ellps=GRS80")
    with pytest.raises(mp.CRSError, match="parameters"):
        mp.CRS("+proj=utm +zone=18 +k=0.5")
    with pytest.raises(mp.CRSError, match="spherical or WGS84"):
        mp.CRS("+proj=merc +a=6378137 +b=6000000")


def test_proj_forward_inverse_parity():
    ours = mp.Proj("EPSG:32618")
    reference = pyproj.Proj("EPSG:32618")
    xy = ours(-75.0, 40.0)
    assert xy == pytest.approx(reference(-75.0, 40.0), abs=2e-4)
    assert ours(*xy, inverse=True) == pytest.approx(
        reference(*xy, inverse=True), abs=1.1e-8
    )
    assert not ours.is_latlong()
    assert not ours.is_geocent()


def test_top_level_transform_and_itransform():
    source = mp.Proj("EPSG:4326")
    target = mp.Proj("EPSG:3857")
    expected = mp.Transformer.from_crs(4326, 3857, always_xy=True).transform(
        [-75.0, 2.0], [40.0, 48.0]
    )
    assert np.allclose(
        mp.transform(source, target, [-75.0, 2.0], [40.0, 48.0], always_xy=True),
        expected,
    )
    points = list(mp.itransform(source, target, [(-75.0, 40.0), (2.0, 48.0)], always_xy=True))
    assert np.allclose(points, np.column_stack(expected))


def test_itransform_3d_4d_switch_and_time_third():
    transformer = mp.Transformer.from_crs(4326, 3857, always_xy=True)
    xyz = list(transformer.itransform([(-75.0, 40.0, 12.0)]))
    assert xyz[0][2] == 12.0
    xyt = list(
        transformer.itransform(
            [(40.0, -75.0, 2025.5)], switch=True, time_3rd=True
        )
    )
    assert xyt[0][2] == 2025.5
    xyzt = list(transformer.itransform([(-75.0, 40.0, 12.0, 2025.5)]))
    assert xyzt[0][2:] == (12.0, 2025.5)
    with pytest.raises(mp.ProjError, match="consistently"):
        list(transformer.itransform([(-75.0, 40.0), (-74.0, 41.0, 1.0)]))


def test_transform_time_passthrough():
    transformer = mp.Transformer.from_crs(4979, 4978, always_xy=True)
    result = transformer.transform(-75.0, 40.0, 100.0, 2025.5)
    assert len(result) == 4
    assert result[-1] == 2025.5
