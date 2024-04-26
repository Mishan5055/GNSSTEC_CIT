from f_class import EXPERIMENT, INPUT, SETTING, XYZ, BLH
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import os
from function import spH, spB, spL, __zenith__
from Reader import Read_X
from tqdm import tqdm
import datetime
import iri2016
from scipy.interpolate import RegularGridInterpolator
import math

########################################################################
################### FOR COEFFICIENT MATRIX C ###########################


def __Calc_IPP__(_input: INPUT, iobs: int, _setting: SETTING) -> list[tuple]:

    sat_XYZ = _input.satF[iobs]
    rec_XYZ = _input.recF[iobs]
    sat_BLH = sat_XYZ.to_BLH()
    rec_BLH = rec_XYZ.to_BLH()

    n_all, n_h, n_b, n_l = _setting.nbrock()
    a1h, a1b, a1l = _setting.plains()

    M_H = _setting.M_H
    M_B = _setting.M_B
    M_L = _setting.M_L

    cp = {}

    r_H = rec_BLH.h
    r_B = rec_BLH.b
    s_H = sat_BLH.h
    s_B = sat_BLH.b

    tsat, tmpsat, loop = spH(rec_XYZ,
                             sat_XYZ,
                             M_H,
                             t=(M_H - r_H) / (s_H - r_H))

    for ib in range(n_b + 1):
        ttmp = (a1b[ib] - r_B) / (M_B - r_B)
        tipp, ipp, loop = spB(rec_XYZ, tmpsat, a1b[ib], t=ttmp)
        if abs(ipp.to_BLH().b - a1b[ib]) > 0.1:
            pass
        else:
            cp[tipp * tsat] = ipp
    for il in range(n_l + 1):
        tipp, ipp = spL(rec_XYZ, sat_XYZ, a1l[il])
        if abs(ipp.to_BLH().l - a1l[il]) > 0.1:
            pass
        else:
            cp[tipp] = ipp
    for ih in range(n_h + 1):
        ttmp = (a1h[ih] - r_H) / (M_H - r_H)
        tipp, ipp, loop = spH(rec_XYZ, tmpsat, a1h[ih], t=ttmp)
        if abs(ipp.to_BLH().h - a1h[ih]) > 1.0:
            pass
        else:
            cp[tipp * tsat] = ipp

    cp_sort = sorted(cp.items())
    return cp_sort


def Gen_one_C(cp_sort: list[tuple],
              _setting: SETTING) -> tuple[list[int], list[float]]:

    A_idx = []
    A_data = []

    a1h, a1b, a1l = _setting.plains()
    n_all, n_h, n_b, n_l = _setting.nbrock()
    L = len(cp_sort)
    for i in range(L - 1):
        bef_t = cp_sort[i][0]
        aft_t = cp_sort[i + 1][0]
        if bef_t > 1.005 or aft_t < -0.005:
            pass
        else:
            bef_pos: XYZ = cp_sort[i][1]
            aft_pos: XYZ = cp_sort[i + 1][1]
            mid_pos = bef_pos * 0.5 + aft_pos * 0.5
            mid_BLH = mid_pos.to_BLH()
            hidx = np.searchsorted(a1h, mid_BLH.h) - 1
            bidx = np.searchsorted(a1b, mid_BLH.b) - 1
            lidx = np.searchsorted(a1l, mid_BLH.l) - 1
            if -1 < hidx < n_h and -1 < bidx < n_b and -1 < lidx < n_l:
                idx = lidx + bidx * n_l + hidx * n_b * n_l
                A_idx.append(idx)
                A_data.append((bef_pos - aft_pos).L2())
    return A_idx, A_data


def Gen_A(_drive: str, _input: INPUT, _setting: SETTING) -> lil_matrix:

    n_obs = _input.n_F
    n_all, n_h, n_b, n_l = _setting.nbrock()

    A = lil_matrix((n_obs, n_all))

    for iobs in tqdm(range(n_obs)):
        # 各平面との交点、パラメータtの値のリスト
        cp_sort = __Calc_IPP__(_input, iobs, _setting)
        # A中の列番号と距離
        A_idx, A_data = Gen_one_C(cp_sort, _setting)
        for jidx in range(len(A_idx)):
            A[iobs, A_idx[jidx]] = A_data[jidx]

    return A


########################################################################
########################################################################

########################################################################
################### FOR DELETING PLASMASPHERIC #########################


def Import_iri2016(day_str, M_H, m_B, M_B, m_L, M_L, nb, nl):
    _b = np.linspace(m_B, M_B, nb)
    _l = np.linspace(m_L, M_L, nl)
    ls, bs = np.meshgrid(_l, _b)

    ans = np.full((nl, nb), 0.0, dtype=float)

    for il in range(nl):
        for jb in range(nb):
            ans[il, jb] = iri2016.IRI(day_str, [M_H, M_H, 1], bs[il, jb],
                                      ls[il, jb]).ne[0] * 1.0e+3  #[/(km*m^2)]
    return ans


def Gen_B(exp: EXPERIMENT, _input: INPUT, _setting: SETTING):

    year4 = exp.y
    doy = exp.d
    epoch = exp.ep

    UT = epoch / 120.0
    tmp = datetime.datetime(year=year4, month=1, day=1) + datetime.timedelta(
        days=doy - 1, hours=UT)
    year = tmp.year
    month = tmp.month
    day = tmp.day
    hour = tmp.hour
    minute = tmp.minute
    second = tmp.second
    day_str = "{y:04d}-{m:02d}-{d:02d} {h:02d}:{mn:02d}:{s:02d}".format(
        y=year, m=month, d=day, h=hour, mn=minute, s=second)

    m_H, M_H, m_B, M_B, m_L, M_L = _setting.bounds()
    tecs = _input.tecF
    recs = _input.recF
    sats = _input.satF

    n_obs = _input.n_F

    nb = 7
    nl = 7
    _dIRI = Import_iri2016(day_str, M_H, m_B, M_B, m_L, M_L, nb, nl)

    _b = np.linspace(m_B, M_B, nb)
    _l = np.linspace(m_L, M_L, nl)
    interp = RegularGridInterpolator((_l, _b), _dIRI)

    tecs_copy = tecs.copy()
    scale = 1.0e+3
    for iobs in range(n_obs):
        theta = __zenith__(recs[iobs], sats[iobs], M_H)
        t, IPP, loop = spH(recs[iobs], sats[iobs], M_H)
        dtec = interp([IPP.to_BLH().l, IPP.to_BLH().b])[0] * scale / (
            1.0e+16 * abs(math.cos(theta)))  #[TECU/km]
        tecs_copy[iobs] -= dtec

    # print(1.4, exp)
    return tecs_copy


########################################################################
################### FOR GENRATING H ####################################


def Gen_H(_setting: SETTING):
    n_all, n_h, n_b, n_l = _setting.nbrock()
    alpha = _setting.alpha
    a2h, a2b, a2l = _setting.centers()
    cof = _setting.cof
    H = lil_matrix((n_all, n_all))

    for ih in range(n_h):
        for jb in range(n_b):
            for kl in range(n_l):
                idx = kl + jb * n_l + ih * n_l * n_b
                coeff_sum = 0.0
                # west
                if kl != 0:
                    idx1 = kl - 1 + jb * n_l + ih * n_l * n_b
                    H[idx, idx1] = cof[ih] * -1.0 * alpha
                    coeff_sum -= H[idx, idx1]
                # east
                if kl != n_l - 1:
                    idx2 = kl + 1 + jb * n_l + ih * n_l * n_b
                    H[idx, idx2] = cof[ih] * -1.0 * alpha
                    coeff_sum -= H[idx, idx2]
                # south
                if jb != 0:
                    idx3 = kl + (jb - 1) * n_l + ih * n_l * n_b
                    H[idx, idx3] = cof[ih] * -1.0 * alpha
                    coeff_sum -= H[idx, idx3]
                # north
                if jb != n_b - 1:
                    idx4 = kl + (jb + 1) * n_l + ih * n_l * n_b
                    H[idx, idx4] = cof[ih] * -1.0 * alpha
                    coeff_sum -= H[idx, idx4]
                # below
                if ih != 0:
                    idx5 = kl + jb * n_l + (ih - 1) * n_l * n_b
                    H[idx, idx5] = cof[ih - 1] * -1.0
                    coeff_sum -= H[idx, idx5]
                # above
                if ih < n_h - 1:
                    idx6 = kl + jb * n_l + (ih + 1) * n_l * n_b
                    H[idx, idx6] = cof[ih + 1] * -1.0
                    coeff_sum -= H[idx, idx6]
                if ih == 0 or ih == n_h - 1:
                    H[idx, idx] = coeff_sum + cof[ih] * 1.0
                else:
                    H[idx, idx] = coeff_sum

    return H.tocsr()


########################################################################
################### FOR GENRATING Y ####################################


def Gen_Y(drive, exp: EXPERIMENT, _setting: SETTING):
    n_all, n_h, n_b, n_l = _setting.nbrock()
    a1h, a1b, a1l = _setting.plains()
    a2h, a2b, a2l = _setting.centers()
    m_H, M_H, m_B, M_B, m_L, M_L = _setting.bounds()

    year4 = exp.y
    doy = exp.d
    epoch = exp.ep

    Y = np.full((n_all, 1), 0.0, dtype=float)

    UT = epoch / 120.0
    tmp = datetime.datetime(year=year4, month=1, day=1) + datetime.timedelta(
        days=doy - 1, hours=UT)
    year = tmp.year
    month = tmp.month
    day = tmp.day
    hour = tmp.hour
    minute = tmp.minute
    second = tmp.second
    sday = "{y:04d}-{m:02d}-{d:02d} {h:02d}:{mn:02d}:{s:02d}".format(y=year,
                                                                     m=month,
                                                                     d=day,
                                                                     h=hour,
                                                                     mn=minute,
                                                                     s=second)
    nb = 7
    nl = 7
    _dIRI = Import_iri2016(sday, M_H, m_B, M_B, m_L, M_L, nb, nl)

    _b = np.linspace(m_B, M_B, nb)
    _l = np.linspace(m_L, M_L, nl)
    interp = RegularGridInterpolator((_l, _b), _dIRI)

    for jb in range(n_b):
        for kl in range(n_l):
            idx1 = kl + jb * n_l + (n_h - 1) * n_l * n_b
            Y[idx1, 0] = interp([a2l[kl], a2b[jb]]) / 1.0e+16

    return Y


def Gen_X0(drive, bef_exps: list[EXPERIMENT], setting: SETTING, _deg=1):
    """
    bef_exps[0]は初期解X0を生成したい時刻を指定してください。それ以降の時刻は解が得られている時刻を指定してください。
    """
    N = len(bef_exps)
    n_all, n_h, n_b, n_l = setting.nbrock()
    Xs = np.full((N - 1, n_all), np.nan, dtype=float)
    Ts = np.full((N - 1), np.nan, dtype=float)
    ep_0 = bef_exps[0].ep
    for t in range(1, N):
        X0_t = Read_X(drive, bef_exps[t])
        ep_t = bef_exps[t].ep
        Ts[t - 1] = ep_t
        Xs[t - 1, :] = X0_t[:, 0]

    X0 = np.full((n_all, 1), 0.0, dtype=float)
    # print("deg = ", min(_deg, N - 2))

    Z = np.polyfit(Ts, Xs, deg=min(_deg, N - 2))
    for ibox in range(n_all):
        X0[ibox] = np.poly1d(Z[:, ibox])(ep_0)

    return X0
