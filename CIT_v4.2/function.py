import numpy as np
import math
from f_class import BLH, XYZ


def __innerp__(x1: XYZ, x2: XYZ) -> float:
    """_summary_

    Calculate the dot product of two vectors.

    Args:
        x1 (XYZ): _description_
        x2 (XYZ): _description_

    Returns:
        float: _description_
    """
    ans = 0.0
    ans += x1.x * x2.x
    ans += x1.y * x2.y
    ans += x1.z * x2.z
    return ans


def __zenith__(rec: XYZ, sat: XYZ, H: float):
    t, ipp, loop = spH(rec, sat, H)
    cosZ = __innerp__(ipp - rec, ipp) / math.sqrt(
        __innerp__(ipp, ipp) * __innerp__(ipp - rec, ipp - rec))
    return math.acos(cosZ)


def spH(rec: XYZ,
        sat: XYZ,
        H,
        t=1.0,
        eps=0.02,
        loop=0) -> tuple[float, XYZ, int]:
    """_summary_

    受信局(rec)と衛星(sat)の間に通したパスのうち、高度Hである点(XYZ)を返します。
    tはrecでt=0,satでt=1となるように設定されたパラメータ。
    epsは関数が停止する閾値。デフォルトでは0.02km。
    
    
    Args:
        rec (XYZ): _description_
        sat (XYZ): _description_
        H (float): _description_
        t (float, optional): _description_. Defaults to 1.0.
        eps (float, optional): _description_. Defaults to 0.02.

    Returns:
        tuple[float, XYZ]: _description_
    """
    dt = 1.0e-4
    point = XYZ(rec.x * (1 - t) + sat.x * t,
                rec.y * (1 - t) + sat.y * t,
                rec.z * (1 - t) + sat.z * t)
    if abs(point.to_BLH().h - H) < eps:
        if t < 0.0:
            return -1.0, rec, loop
        elif t > 1.0:
            return 2.0, sat, loop
        else:
            return t, point, loop
    else:
        dpoint = XYZ(rec.x * (1 - t - dt) + sat.x * (t + dt),
                     rec.y * (1 - t - dt) + sat.y * (t + dt),
                     rec.z * (1 - t - dt) + sat.z * (t + dt))
        fbar_i = (dpoint.to_BLH().h - point.to_BLH().h) / dt
        tbar = t - (point.to_BLH().h - H) / fbar_i
        return spH(rec, sat, H, tbar, eps, loop=loop + 1)


def spB(rec: XYZ,
        sat: XYZ,
        B,
        t=1.0,
        eps=20e-5,
        loop=0) -> tuple[float, XYZ, int]:
    # 1.0e-5[deg]=1[m]
    dt = 1.0e-5
    point = XYZ(rec.x * (1 - t) + sat.x * t,
                rec.y * (1 - t) + sat.y * t,
                rec.z * (1 - t) + sat.z * t)
    if t < 0.0 or t > 1.0:
        return -1.0, rec, loop
    if abs(point.to_BLH().b - B) < eps:
        if t < 0.0:
            return -1.0, rec, loop
        elif t > 1.0:
            return 2.0, sat, loop
        else:
            return t, point, loop
    else:
        dpoint = XYZ(rec.x * (1 - t - dt) + sat.x * (t + dt),
                     rec.y * (1 - t - dt) + sat.y * (t + dt),
                     rec.z * (1 - t - dt) + sat.z * (t + dt))
        fbar_i = (dpoint.to_BLH().b - point.to_BLH().b) / dt
        tbar = t - (point.to_BLH().b - B) / fbar_i
        return spB(rec, sat, B, tbar, eps, loop=loop + 1)


def spL(rec: XYZ, sat: XYZ, L) -> tuple[float, XYZ]:
    tanL = math.tan(math.radians(L))
    s = (sat.y - sat.x * tanL) / ((rec.x - sat.x) * tanL - rec.y + sat.y)
    ans = XYZ(0.0, 0.0, 0.0)
    ans.x = s * rec.x + (1 - s) * sat.x
    ans.y = s * rec.y + (1 - s) * sat.y
    ans.z = s * rec.z + (1 - s) * sat.z
    return 1.0 - s, ans
