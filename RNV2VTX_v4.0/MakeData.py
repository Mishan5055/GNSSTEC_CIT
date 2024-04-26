import numpy as np
import os
import datetime

from function import gtxyz
from f_class import XYZ

np.set_printoptions(edgeitems=2880)


def __MakeOrbitData__(isat: int, tiempo, oel, i1stsat):
    orbit = np.full((2880, 3), np.nan, dtype=float)

    if len(i1stsat[isat]) > 0:
        base = 0
        for iepoc in range(2880):
            ttime = iepoc / 120.0
            time0 = tiempo[i1stsat[isat][base]]
            try:
                xyz = gtxyz(ttime, time0, oel[i1stsat[isat][base]])
                orbit[iepoc, 0] = xyz.x
                orbit[iepoc, 1] = xyz.y
                orbit[iepoc, 2] = xyz.z
            except:
                # もしgtxyzが機能しなかったら、baseを1つ後のやつにする
                if base + 1 < len(i1stsat[isat]):
                    base += 1
                    iepoc -= 1
                    continue
                else:
                    pass
        return orbit
    else:
        return orbit


def __MakeGLONASSOrbitData__(isat: int, tiempo, oel, i1stsat):

    L = len(i1stsat[isat])

    param = np.full((2880), 0.0, dtype=float)
    orbit = np.full((2880, 3), 0.0, dtype=float)

    dt = 30.0  #[s]
    mu = 398600.44e9  #[m^3/s^2]
    ae = 6378136  #[m]
    J02 = 1082625.7e-9
    omega = 7.292115e-5  #[rad/s]

    N = 60

    for jrec in range(L):
        front_position = np.full((N, 3), np.nan, dtype=float)
        front_velocity = np.full((N, 3), np.nan, dtype=float)
        acceleration = np.full((3), np.nan, dtype=float)

        record = oel[i1stsat[isat][jrec]]
        recordtime = tiempo[i1stsat[isat][jrec]]
        recordepoch = round(recordtime * 120.0)
        front_position[0, 0] = record[0]
        front_position[0, 1] = record[3]
        front_position[0, 2] = record[6]
        front_velocity[0, 0] = record[1]
        front_velocity[0, 1] = record[4]
        front_velocity[0, 2] = record[7]
        acceleration[0] = record[2]
        acceleration[1] = record[5]
        acceleration[2] = record[8]
        for kepoc in range(N - 1):
            r = np.linalg.norm(front_position[kepoc, :], ord=2)
            dVdt = np.full((3), 0.0, dtype=float)
            first = -mu / pow(r, 3)
            second = -1.5 * J02 * mu * pow(ae, 2) * (
                1 - 5 * pow(front_position[kepoc, 2], 2) / pow(r, 2)) / pow(
                    r, 5)
            dVdt += first * front_position[kepoc, :]
            dVdt += second * front_position[kepoc, :]
            dVdt += acceleration
            dVdt[0] += (pow(omega, 2) * front_position[kepoc, 0] +
                        2 * omega * front_velocity[kepoc, 0])
            dVdt[1] += (pow(omega, 2) * front_position[kepoc, 1] +
                        2 * omega * front_velocity[kepoc, 1])
            front_velocity[kepoc + 1] = front_velocity[kepoc] + dt * dVdt
            front_position[
                kepoc + 1] = front_position[kepoc] + dt * front_velocity[kepoc]

        for kepoc in range(N):
            if -1 < recordepoch + kepoc < 2880:
                param[recordepoch + kepoc] = (1.0 - kepoc / N)
                orbit[recordepoch +
                      kepoc, :] += (1.0 - kepoc / N) * front_position[kepoc, :]

        back_position = np.full((N, 3), 0.0, dtype=float)
        back_velocity = np.full((N, 3), 0.0, dtype=float)

        back_position[0, 0] = record[0]
        back_position[0, 1] = record[3]
        back_position[0, 2] = record[6]
        back_velocity[0, 0] = record[1]
        back_velocity[0, 1] = record[4]
        back_velocity[0, 2] = record[7]

        for kepoc in range(N - 1):
            r = np.linalg.norm(back_position[kepoc, :], ord=2)
            dVdt = np.full((3), 0.0, dtype=float)
            first = -mu / pow(r, 3)
            second = -1.5 * J02 * mu * pow(ae, 2) * (
                1 - 5 * pow(back_position[kepoc, 2], 2) / pow(r, 2)) / pow(
                    r, 5)
            dVdt += first * back_position[kepoc, :]
            dVdt += second * back_position[kepoc, :]
            dVdt += acceleration
            dVdt[0] += (pow(omega, 2) * back_position[kepoc, 0] +
                        2 * omega * back_velocity[kepoc, 0])
            dVdt[1] += (pow(omega, 2) * back_position[kepoc, 1] +
                        2 * omega * back_velocity[kepoc, 1])
            back_velocity[kepoc + 1] = back_velocity[kepoc] + (-dt) * dVdt
            back_position[
                kepoc +
                1] = back_position[kepoc] + (-dt) * back_velocity[kepoc]

        for kepoc in range(1, N):
            if -1 < recordepoch - kepoc < 2880:
                param[recordepoch - kepoc] += (1.0 - kepoc / N)
                orbit[recordepoch -
                      kepoc, :] += (1.0 - kepoc / N) * back_position[kepoc, :]

    result = np.full((2880, 3), np.nan, dtype=float)
    for jepoc in range(2880):
        if param[jepoc] > 0.99:
            result[jepoc, :] = orbit[jepoc, :]

    return result


def __MakeMDFFile__(path_nd, mdfLstec, Lstec, Pstec, L_valid, p_o: XYZ, sat_id,
                    orbit, Zs, L1, L2, ver):
    """_summary_

    .mdfファイルを作成し、必要なデータを書き込む関数です。\n
    
    Args:
        path_nd (str): .mdfファイルのパス
        mdfLstec (numpy.ndarray): modified L stec data
        Lstec (numpy.ndarray): L stec data
        Pstec (numpy.ndarray): P stec data
        L_valid (numpy.ndarray): Valid data
        p_o (XYZ): 受信局座標
        sat_id (str): 衛星のPRN ID
        orbit (numpy.ndarray): 衛星軌道
        Zs (numpy.ndarray): 衛星の仰角
        L1 (float or int): L1 frequency (Hz)
        L2 (float or int): L2 frequency (Hz)
        ver (str): RNX2VTX version
    """

    with open(path_nd, "w") as f:
        print(f"# {sat_id}", file=f)
        print(f"#", file=f)
        print(f"# RNX2VTX Version : {ver}", file=f)
        print(f"# {datetime.datetime.now()}", file=f)
        print(f"# ", file=f)
        print("# Receiver Position : {x:014.4f} {y:014.4f} {z:014.4f}".format(
            x=p_o.x, y=p_o.y, z=p_o.z),
              file=f)
        print(f"# ", file=f)
        print(f"# END OF HEADER", file=f)

        for iepoc in range(2880):
            if L_valid[iepoc] > 0:
                print(
                    "{t:07.4f} {mdf:09.4f} {p:09.4f} {l:09.4f} {x:014.4f} {y:014.4f} {z:014.4f} {zen:07.4f}"
                    .format(t=iepoc / 120.0,
                            mdf=mdfLstec[iepoc],
                            p=Pstec[iepoc],
                            l=Lstec[iepoc],
                            x=orbit[iepoc, 0],
                            y=orbit[iepoc, 1],
                            z=orbit[iepoc, 2],
                            zen=90.0 - Zs[iepoc]),
                    file=f)

    return
