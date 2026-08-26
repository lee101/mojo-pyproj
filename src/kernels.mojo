"""Projection and datum-transformation kernels exposed through a C ABI."""

from std.ffi import external_call
from std.math import atan2, cos, exp, pow, sin, sqrt, tan
from std.sys.info import num_physical_cores, simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime PI = 3.141592653589793238462643383279502884
comptime DEG = PI / 180.0
comptime RAD = 180.0 / PI
comptime A = 6378137.0
comptime INV_F = 298.257223563
comptime F = 1.0 / INV_F
comptime E2 = F * (2.0 - F)
comptime EP2 = E2 / (1.0 - E2)
comptime B = A * (1.0 - F)
comptime THIRD_FLATTENING = F / (2.0 - F)
comptime K0 = 0.9996
comptime PARALLEL_POINTS = 65536
comptime HELMERT_PARALLEL_POINTS = 262144
comptime HELMERT_MAX_WORKERS = 8
comptime W = simd_width_of[DType.float64]()


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def workers_for(n: Int) -> Int:
    return min(n, num_physical_cores()) if n >= PARALLEL_POINTS else 1


def helmert_workers_for(n: Int) -> Int:
    return (
        min(n, min(num_physical_cores(), HELMERT_MAX_WORKERS)) if n
        >= HELMERT_PARALLEL_POINTS else 1
    )


@always_inline
def precise_log(value: Float64) -> Float64:
    return external_call["log", Float64](value)


@always_inline
def meridional_arc(phi: Float64) -> Float64:
    var n = THIRD_FLATTENING
    var n2 = n * n
    var n3 = n2 * n
    var n4 = n2 * n2
    var n5 = n4 * n
    var n6 = n3 * n3
    var a0 = 1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0
    var a2 = 1.5 * (n - n3 / 8.0 - n5 / 64.0)
    var a4 = 15.0 / 16.0 * (n2 - n4 / 4.0 - n6 / 16.0)
    var a6 = 35.0 / 48.0 * (n3 - 5.0 * n5 / 16.0)
    var a8 = 315.0 / 512.0 * (n4 - 3.0 * n6 / 8.0)
    var a10 = 693.0 / 1280.0 * n5
    var a12 = 1001.0 / 2048.0 * n6
    return (
        A
        / (1.0 + n)
        * (
            a0 * phi
            - a2 * sin(2.0 * phi)
            + a4 * sin(4.0 * phi)
            - a6 * sin(6.0 * phi)
            + a8 * sin(8.0 * phi)
            - a10 * sin(10.0 * phi)
            + a12 * sin(12.0 * phi)
        )
    )


@export("mpj_web_mercator")
def web_mercator(
    x_addr: Int,
    y_addr: Int,
    ox_addr: Int,
    oy_addr: Int,
    n: Int,
    inverse: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var ox = p(ox_addr)
    var oy = p(oy_addr)
    var workers = workers_for(n)

    @__parameter
    def process(worker: Int):
        var start = worker * n // workers
        var stop = (worker + 1) * n // workers
        for i in range(start, stop):
            if inverse == 0:
                var phi = y[unsafe_offset=i] * DEG
                var s = sin(phi)
                ox[unsafe_offset=i] = A * x[unsafe_offset=i] * DEG
                oy[unsafe_offset=i] = (
                    0.5 * A * precise_log((1.0 + s) / (1.0 - s))
                )
            else:
                ox[unsafe_offset=i] = x[unsafe_offset=i] / A * RAD
                oy[unsafe_offset=i] = (
                    2.0 * atan2(exp(y[unsafe_offset=i] / A), 1.0) - PI * 0.5
                ) * RAD

    for worker in range(workers):
        process(worker)


@export("mpj_world_mercator")
def world_mercator(
    x_addr: Int,
    y_addr: Int,
    ox_addr: Int,
    oy_addr: Int,
    n: Int,
    inverse: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var ox = p(ox_addr)
    var oy = p(oy_addr)
    var workers = workers_for(n)
    var eccentricity = sqrt(E2)

    @__parameter
    def process(worker: Int):
        var start = worker * n // workers
        var stop = (worker + 1) * n // workers
        for i in range(start, stop):
            if inverse == 0:
                var phi = y[unsafe_offset=i] * DEG
                var s = sin(phi)
                ox[unsafe_offset=i] = A * x[unsafe_offset=i] * DEG
                oy[unsafe_offset=i] = (
                    0.5
                    * A
                    * (
                        precise_log((1.0 + s) / (1.0 - s))
                        - eccentricity
                        * precise_log(
                            (1.0 + eccentricity * s) / (1.0 - eccentricity * s)
                        )
                    )
                )
            else:
                ox[unsafe_offset=i] = x[unsafe_offset=i] / A * RAD
                var t = exp(-y[unsafe_offset=i] / A)
                var phi = PI * 0.5 - 2.0 * atan2(t, 1.0)
                for _ in range(8):
                    var es = eccentricity * sin(phi)
                    phi = PI * 0.5 - 2.0 * atan2(
                        t * pow((1.0 - es) / (1.0 + es), 0.5 * eccentricity),
                        1.0,
                    )
                oy[unsafe_offset=i] = phi * RAD

    for worker in range(workers):
        process(worker)


@export("mpj_utm")
def utm(
    x_addr: Int,
    y_addr: Int,
    ox_addr: Int,
    oy_addr: Int,
    n: Int,
    zone: Int,
    south: Int,
    inverse: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var ox = p(ox_addr)
    var oy = p(oy_addr)
    var workers = workers_for(n)
    var lon0 = (Float64(zone) * 6.0 - 183.0) * DEG

    @__parameter
    def process(worker: Int):
        var start = worker * n // workers
        var stop = (worker + 1) * n // workers
        for i in range(start, stop):
            if inverse == 0:
                var lon = x[unsafe_offset=i] * DEG
                var phi = y[unsafe_offset=i] * DEG
                var sin_phi = sin(phi)
                var cos_phi = cos(phi)
                var tan_phi = tan(phi)
                var tan2 = tan_phi * tan_phi
                var c = EP2 * cos_phi * cos_phi
                var aa = cos_phi * (lon - lon0)
                var aa2 = aa * aa
                var aa3 = aa2 * aa
                var aa4 = aa2 * aa2
                var aa5 = aa4 * aa
                var aa6 = aa3 * aa3
                var nu = A / sqrt(1.0 - E2 * sin_phi * sin_phi)
                ox[unsafe_offset=i] = 500000.0 + K0 * nu * (
                    aa
                    + (1.0 - tan2 + c) * aa3 / 6.0
                    + (5.0 - 18.0 * tan2 + tan2 * tan2 + 72.0 * c - 58.0 * EP2)
                    * aa5
                    / 120.0
                )
                var northing = K0 * (
                    meridional_arc(phi)
                    + nu
                    * tan_phi
                    * (
                        aa2 / 2.0
                        + (5.0 - tan2 + 9.0 * c + 4.0 * c * c) * aa4 / 24.0
                        + (
                            61.0
                            - 58.0 * tan2
                            + tan2 * tan2
                            + 600.0 * c
                            - 330.0 * EP2
                        )
                        * aa6
                        / 720.0
                    )
                )
                oy[unsafe_offset=i] = northing + (
                    10000000.0 if south != 0 else 0.0
                )
            else:
                var east = x[unsafe_offset=i] - 500000.0
                var north = y[unsafe_offset=i] - (
                    10000000.0 if south != 0 else 0.0
                )
                var m = north / K0
                var phi1 = m / A
                for _ in range(5):
                    var foot_sin = sin(phi1)
                    var denom = 1.0 - E2 * foot_sin * foot_sin
                    var rho = A * (1.0 - E2) / pow(denom, 1.5)
                    phi1 += (m - meridional_arc(phi1)) / rho
                var sp = sin(phi1)
                var cp = cos(phi1)
                var tp = tan(phi1)
                var t1 = tp * tp
                var c1 = EP2 * cp * cp
                var n1 = A / sqrt(1.0 - E2 * sp * sp)
                var r1 = A * (1.0 - E2) / pow(1.0 - E2 * sp * sp, 1.5)
                var d = east / (n1 * K0)
                var d2 = d * d
                var d3 = d2 * d
                var d4 = d2 * d2
                var d5 = d4 * d
                var d6 = d3 * d3
                var lat = phi1 - (n1 * tp / r1) * (
                    d2 / 2.0
                    - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * EP2)
                    * d4
                    / 24.0
                    + (
                        61.0
                        + 90.0 * t1
                        + 298.0 * c1
                        + 45.0 * t1 * t1
                        - 252.0 * EP2
                        - 3.0 * c1 * c1
                    )
                    * d6
                    / 720.0
                )
                var lon = (
                    lon0
                    + (
                        d
                        - (1.0 + 2.0 * t1 + c1) * d3 / 6.0
                        + (
                            5.0
                            - 2.0 * c1
                            + 28.0 * t1
                            - 3.0 * c1 * c1
                            + 8.0 * EP2
                            + 24.0 * t1 * t1
                        )
                        * d5
                        / 120.0
                    )
                    / cp
                )
                ox[unsafe_offset=i] = lon * RAD
                oy[unsafe_offset=i] = lat * RAD

    for worker in range(workers):
        process(worker)


@export("mpj_geocentric")
def geocentric(
    x_addr: Int,
    y_addr: Int,
    z_addr: Int,
    ox_addr: Int,
    oy_addr: Int,
    oz_addr: Int,
    n: Int,
    inverse: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var z = p(z_addr)
    var ox = p(ox_addr)
    var oy = p(oy_addr)
    var oz = p(oz_addr)
    var workers = workers_for(n)

    @__parameter
    def process(worker: Int):
        var start = worker * n // workers
        var stop = (worker + 1) * n // workers
        for i in range(start, stop):
            if inverse == 0:
                var lon = x[unsafe_offset=i] * DEG
                var lat = y[unsafe_offset=i] * DEG
                var slat = sin(lat)
                var clat = cos(lat)
                var nu = A / sqrt(1.0 - E2 * slat * slat)
                ox[unsafe_offset=i] = (
                    (nu + z[unsafe_offset=i]) * clat * cos(lon)
                )
                oy[unsafe_offset=i] = (
                    (nu + z[unsafe_offset=i]) * clat * sin(lon)
                )
                oz[unsafe_offset=i] = (
                    nu * (1.0 - E2) + z[unsafe_offset=i]
                ) * slat
            else:
                var xx = x[unsafe_offset=i]
                var yy = y[unsafe_offset=i]
                var zz = z[unsafe_offset=i]
                var radius = sqrt(xx * xx + yy * yy)
                var theta = atan2(zz * A, radius * B)
                var st = sin(theta)
                var ct = cos(theta)
                var lat = atan2(
                    zz + EP2 * B * st * st * st,
                    radius - E2 * A * ct * ct * ct,
                )
                var height: Float64
                for _ in range(3):
                    var slat = sin(lat)
                    var nu = A / sqrt(1.0 - E2 * slat * slat)
                    height = radius / cos(lat) - nu
                    lat = atan2(
                        zz,
                        radius * (1.0 - E2 * nu / (nu + height)),
                    )
                var final_sin = sin(lat)
                var final_nu = A / sqrt(1.0 - E2 * final_sin * final_sin)
                height = radius / cos(lat) - final_nu
                ox[unsafe_offset=i] = atan2(yy, xx) * RAD
                oy[unsafe_offset=i] = lat * RAD
                oz[unsafe_offset=i] = height

    for worker in range(workers):
        process(worker)


@export("mpj_helmert")
def helmert(
    x_addr: Int,
    y_addr: Int,
    z_addr: Int,
    ox_addr: Int,
    oy_addr: Int,
    oz_addr: Int,
    n: Int,
    tx: Float64,
    ty: Float64,
    tz: Float64,
    rx_arcsec: Float64,
    ry_arcsec: Float64,
    rz_arcsec: Float64,
    scale_ppm: Float64,
    convention_sign: Float64,
    inverse: Int,
) abi("C"):
    var x = p(x_addr)
    var y = p(y_addr)
    var z = p(z_addr)
    var ox = p(ox_addr)
    var oy = p(oy_addr)
    var oz = p(oz_addr)
    var workers = helmert_workers_for(n)
    var arcsec = DEG / 3600.0
    var rx = convention_sign * rx_arcsec * arcsec
    var ry = convention_sign * ry_arcsec * arcsec
    var rz = convention_sign * rz_arcsec * arcsec
    var scale = 1.0 + scale_ppm * 1.0e-6

    @__parameter
    def process(worker: Int):
        var start = worker * n // workers
        var stop = (worker + 1) * n // workers
        var vector_stop = start + (stop - start) // W * W
        if inverse == 0:
            for i in range(start, vector_stop, W):
                var xv = x.unsafe_load[width=W](i)
                var yv = y.unsafe_load[width=W](i)
                var zv = z.unsafe_load[width=W](i)
                ox.unsafe_store(i, tx + scale * (xv - rz * yv + ry * zv))
                oy.unsafe_store(i, ty + scale * (rz * xv + yv - rx * zv))
                oz.unsafe_store(i, tz + scale * (-ry * xv + rx * yv + zv))
            for i in range(vector_stop, stop):
                ox[unsafe_offset=i] = tx + scale * (
                    x[unsafe_offset=i]
                    - rz * y[unsafe_offset=i]
                    + ry * z[unsafe_offset=i]
                )
                oy[unsafe_offset=i] = ty + scale * (
                    rz * x[unsafe_offset=i]
                    + y[unsafe_offset=i]
                    - rx * z[unsafe_offset=i]
                )
                oz[unsafe_offset=i] = tz + scale * (
                    -ry * x[unsafe_offset=i]
                    + rx * y[unsafe_offset=i]
                    + z[unsafe_offset=i]
                )
        else:
            for i in range(start, vector_stop, W):
                var xx = (x.unsafe_load[width=W](i) - tx) / scale
                var yy = (y.unsafe_load[width=W](i) - ty) / scale
                var zz = (z.unsafe_load[width=W](i) - tz) / scale
                ox.unsafe_store(i, xx + rz * yy - ry * zz)
                oy.unsafe_store(i, -rz * xx + yy + rx * zz)
                oz.unsafe_store(i, ry * xx - rx * yy + zz)
            for i in range(vector_stop, stop):
                var xx = (x[unsafe_offset=i] - tx) / scale
                var yy = (y[unsafe_offset=i] - ty) / scale
                var zz = (z[unsafe_offset=i] - tz) / scale
                ox[unsafe_offset=i] = xx + rz * yy - ry * zz
                oy[unsafe_offset=i] = -rz * xx + yy + rx * zz
                oz[unsafe_offset=i] = ry * xx - rx * yy + zz

    for worker in range(workers):
        process(worker)
