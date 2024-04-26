from f_class import XYZ
import math


def __innerp(p1: XYZ, p2: XYZ) -> float:
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z


def specify_H(rec: XYZ, sat: XYZ, H, t=1.0) -> tuple[float, XYZ]:
    # 1[m]
    alpha = 1.0
    eps = 0.001
    dt = 0.01
    point = XYZ(
        rec.x * (1 - t) + sat.x * t,
        rec.y * (1 - t) + sat.y * t,
        rec.z * (1 - t) + sat.z * t,
    )
    if abs(point.to_BLH().h - H) < eps:
        if t < 0.0 or t > 1.0:
            return t, rec
        else:
            return t, point
    else:
        dpoint = XYZ(
            rec.x * (1 - t - dt) + sat.x * (t + dt),
            rec.y * (1 - t - dt) + sat.y * (t + dt),
            rec.z * (1 - t - dt) + sat.z * (t + dt),
        )
        fbar_i = (dpoint.to_BLH().h - point.to_BLH().h) / dt
        tbar = t - alpha * (point.to_BLH().h - H) / fbar_i
        # print(t, tbar)
        return specify_H(rec, sat, H, tbar)


def sat_zenith(rec: XYZ, sat: XYZ) -> float:
    """
    Return zenith angle(radians) between rec(XYZ) to sat(XYZ) at rec.
    """
    r2s = sat - rec
    cosZ = __innerp(rec, r2s) / (rec.L2() * r2s.L2())
    return math.acos(cosZ)


def ipp_zenith(rec: XYZ, sat: XYZ, H: float) -> float:
    """
    Return zenith angle(radians) between rec(XYZ) to sat(XYZ) at height H[m].
    """
    t, ipp = specify_H(rec, sat, H)
    r2s = sat - rec
    cosZ = __innerp(ipp, r2s) / (ipp.L2() * r2s.L2())
    return math.acos(cosZ)


def kepler(dmk, e):
    thres = pow(10, -14)
    niteration = 0
    ek = dmk
    diff = ek + e * math.sin(ek) - dmk  # [rad]
    while abs(diff) > thres:
        diff = ek + e * math.sin(ek) - dmk
        partial = 1 - e * math.cos(ek)
        ek = ek - diff / partial
        niteration += 1
        if niteration > 100:
            print("The calculation was terminated because the iteration of Newton's method in the Kep1er function exceeded 100.")
            break
    return ek


def gtxyz(time1, time0, ele):
    # data format of navigation file
    #       0:IODE  1:Crs  2:delta-n  3:m0
    #       4:Cuc   5:e    6:Cus      7:root-a
    #       8:Toe   9:Cic 10:Omega   11:Cis
    #      12:i0   13:Crc 14:omega   15:OmegaDot
    #      16:iDot 17-27: not used
    GM = 3.986005 * math.pow(10, 14)  # [m^3/s^2]
    omega_dot_e = 7.292115 * math.pow(10, -5)  # [rad/s]
    # (1)
    # print(ele[7])
    a = ele[7] ** 2  # [m]
    # (2)
    dnzero = math.sqrt(GM * math.pow(a, -3.0))  # [rad/s]
    # (3)
    tk = (time1 - time0) * 60.0 * 60.0  # [s]
    # (4)
    dn = dnzero + ele[2]  # [rad/s]
    # (5)
    dmk = ele[3] + dn * tk  # [rad]
    # (6)
    ek = kepler(dmk, ele[5])  # [rad]
    # (7)
    cosvk = (math.cos(ek) - ele[5]) / (1.0 - ele[5] * math.cos(ek))
    sinvk = math.sqrt(1.0 - ele[5] * ele[5]) * math.sin(ek) / (1.0 - ele[5] * math.cos(ek))
    vk = math.atan2(sinvk, cosvk)  # [rad]
    # (8)
    phik = vk + ele[14]  # [rad]
    # (9)
    delta_uk = ele[6] * math.sin(2.0 * phik) + ele[4] * math.cos(2.0 * phik)  # [rad]
    uk = phik + delta_uk  # [rad]
    # (10)
    delta_rk = ele[1] * math.sin(2.0 * phik) + ele[13] * math.cos(2.0 * phik)  # [m]
    rk = a * (1.0 - ele[5] * math.cos(ek)) + delta_rk
    # (11)
    delta_dik = ele[11] * math.sin(2.0 * phik) + ele[9] * math.cos(2.0 * phik)
    dik = ele[12] + delta_dik + ele[16] * tk  # [rad]
    # (12)
    xdashk = rk * math.cos(uk)  # [m]
    ydashk = rk * math.sin(uk)  # [m]
    # (13)
    # [rad]    [rad]   [rad/s]    [rad/2]   [s]   [s]   [rad/s]
    omegak = ele[10] + (ele[15] - omega_dot_e) * tk - ele[8] * omega_dot_e
    # (14)
    dx = xdashk * math.cos(omegak) - ydashk * math.cos(dik) * math.sin(omegak)
    dy = xdashk * math.sin(omegak) + ydashk * math.cos(dik) * math.cos(omegak)
    dz = ydashk * math.sin(dik)
    return XYZ(dx, dy, dz)


if __name__ == "__main__":
    rec = XYZ()
