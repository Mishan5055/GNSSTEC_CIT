import datetime
import numpy as np
import time
import os
import math
from tqdm import tqdm

from f_class import XYZ
from function import sat_zenith, specify_H, ipp_zenith
from ReadObs import __ReadObsFile302__
from ReadNav import __ReadNavFile302__, __ReadGLONASSNavFile302__
from MakeData import __MakeOrbitData__, __MakeMDFFile__, __MakeGLONASSOrbitData__

Mrecord = 0
Msat = 0
MGsat = 0
MRsat = 0
MEsat = 0
MJsat = 0

Zenith_Threshold = 0.0
Null_Threshold = 0
TEC_Threshold = 0.0
Valid_Data_Length = 0

H_IPP = 0.0

G1_freq = 0
G2_freq = 0
R1_basefreq = 0
R2_basefreq = 0
R1_deltafreq = 0
R2_deltafreq = 0
E1_freq = 0
E2_freq = 0
J1_freq = 0
J2_freq = 0

GLONASS_SLOT = {}


def __setfile__(file):
    global Mrecord, Msat
    global MGsat, MRsat, MEsat, MJsat
    global G1_freq, G2_freq
    global R1_basefreq, R1_deltafreq, R2_basefreq, R2_deltafreq
    global E1_freq, E2_freq
    global J1_freq, J2_freq
    global H_IPP
    global Zenith_Threshold, Null_Threshold, TEC_Threshold
    global Valid_Data_Length
    with open(file, "r") as f:
        while True:
            line = f.readline()

            if not line:
                break

            if "MAX RECORD" in line:
                Mrecord = int(line.split()[0])
            if "MAX SATELLITE" in line:
                Msat = int(line.split()[0])
            if "MAX GPS" in line:
                MGsat = int(line.split()[0])
            if "MAX GLONASS" in line:
                MRsat = int(line.split()[0])
            if "MAX GALILEO" in line:
                MEsat = int(line.split()[0])
            if "MAX QZSS" in line:
                MJsat = int(line.split()[0])

            if "f1GPS" in line:
                G1_freq = int(line.split()[0])
            if "f2GPS" in line:
                G2_freq = int(line.split()[0])
            if "f1GLONASS" in line:
                R1_basefreq = int(line.split()[0])
                R1_deltafreq = int(line.split()[1])
            if "f2GLONASS" in line:
                R2_basefreq = int(line.split()[0])
                R2_deltafreq = int(line.split()[1])
            if "f1GALILEO" in line:
                E1_freq = int(line.split()[0])
            if "f2GALILEO" in line:
                E2_freq = int(line.split()[0])
            if "f1QZSS" in line:
                J1_freq = int(line.split()[0])
            if "f2QZSS" in line:
                J2_freq = int(line.split()[0])
            if "IPP HEIGHT" in line:
                H_IPP = float(line.split()[0])
            if "ZENITH THRESHOLD" in line:
                Zenith_Threshold = float(line.split()[0])
            if "NULL THRESHOLD" in line:
                Null_Threshold = int(line.split()[0])
            if "TEC THRESHOLD" in line:
                TEC_Threshold = float(line.split()[0])
            if "VALID DATA LENGTH" in line:
                Valid_Data_Length = int(line.split()[0])
            if "GLONASS SLOT" in line:
                for i in range(8):
                    sat_id = line[7 * i:3 + 7 * i]
                    slot = line[4 + 7 * i:6 + 7 * i]
                    if not sat_id in GLONASS_SLOT:
                        GLONASS_SLOT[sat_id] = int(slot)


def __RNX2MDF302__(drive: str, country: str, year4: int, day: int,
                   stations: list, setfile: str):
    __setfile__(setfile)
    year2 = year4 % 100
    obsday = datetime.datetime(year=year4, month=1,
                               day=1) + datetime.timedelta(days=day - 1)

    Gorbit = np.full((2880, MGsat, 3), 0.0, dtype=float)
    Gorbit_bool = np.full((2880, MGsat), False, dtype=bool)
    Gfinish = np.full((MGsat), False, dtype=bool)
    Rorbit = np.full((2880, MRsat, 3), 0.0, dtype=float)
    Rorbit_bool = np.full((2880, MRsat), False, dtype=bool)
    Rfinish = np.full((MRsat), False, dtype=bool)
    Eorbit = np.full((2880, MEsat, 3), 0.0, dtype=float)
    Eorbit_bool = np.full((2880, MEsat), False, dtype=bool)
    Efinish = np.full((MEsat), False, dtype=bool)
    Jorbit = np.full((2880, MJsat, 3), 0.0, dtype=float)
    Jorbit_bool = np.full((2880, MJsat), False, dtype=bool)
    Jfinish = np.full((MJsat), False, dtype=bool)

    start = time.time()

    print("Satellite orbit data is being acquired...")

    for sta in tqdm(stations):

        if not np.all(Gfinish):
            path_n = "{dr}/tec/{c}/{y4:04d}/{d:03d}/{s}{d:03d}0.{y2:02}n".format(
                dr=drive, c=country, y4=year4, d=day, s=sta, y2=year2)

            if os.path.exists(path_n):  ## ここから 12/08
                G_tiempo, G_oel, G_i1stsat = __ReadNavFile302__(
                    path_n, obsday, setfile)

                for isat in range(1, MGsat):
                    if not Gfinish[isat]:
                        Gor_data = __MakeOrbitData__(isat, G_tiempo, G_oel,
                                                     G_i1stsat)
                        for iepoc in range(2880):
                            if not np.isnan(Gor_data[iepoc, 0]):
                                Gorbit[iepoc, isat, :] = Gor_data[iepoc, :]
                                Gorbit_bool[iepoc, isat] = True

                    if (not Gfinish[isat]) or np.all(Gorbit_bool[:, isat]):
                        Gfinish[isat] = True

        if not np.all(Rfinish):
            path_g = "{dr}/tec/{c}/{y4:04d}/{d:03d}/{s}{d:03d}0.{y2:02}g".format(
                dr=drive, c=country, y4=year4, d=day, s=sta, y2=year2)

            if os.path.exists(path_g):
                R_tiempo, R_oel, R_i1stsat = __ReadGLONASSNavFile302__(
                    path_g, obsday, setfile)

                for isat in range(1, MRsat):
                    if not Rfinish[isat]:
                        Ror_data = __MakeGLONASSOrbitData__(
                            isat, R_tiempo, R_oel, R_i1stsat)

                        for iepoc in range(2880):
                            if not np.isnan(Ror_data[iepoc, 0]):
                                Rorbit[iepoc, isat, :] = Ror_data[iepoc, :]
                                Rorbit_bool[iepoc, isat] = True

                    if (not Rfinish[isat]) or np.all(Rorbit_bool[:, isat]):
                        Rfinish[isat] = True

        if not np.all(Efinish):
            path_l = "{dr}/tec/{c}/{y4:04d}/{d:03d}/{s}{d:03d}0.{y2:02}l".format(
                dr=drive, c=country, y4=year4, d=day, s=sta, y2=year2)

            if os.path.exists(path_l):
                E_tiempo, E_oel, E_i1stsat = __ReadNavFile302__(
                    path_l, obsday, setfile)
                for isat in range(1, MEsat):
                    if not Efinish[isat]:
                        Eor_data = __MakeOrbitData__(isat, E_tiempo, E_oel,
                                                     E_i1stsat)
                        for iepoc in range(2880):
                            if not np.isnan(Eor_data[iepoc, 0]):
                                Eorbit[iepoc, isat, :] = Eor_data[iepoc, :]
                                Eorbit_bool[iepoc, isat] = True

                    if np.all(Eorbit_bool[:, isat]):
                        Efinish[isat] = True

        if not np.all(Jfinish):
            path_q = "{dr}/tec/{c}/{y4:04d}/{d:03d}/{s}{d:03d}0.{y2:02}q".format(
                dr=drive, c=country, y4=year4, d=day, s=sta, y2=year2)

            if os.path.exists(path_q):
                J_tiempo, J_oel, J_i1stsat = __ReadNavFile302__(
                    path_q, obsday, setfile)
                for isat in range(1, MJsat):
                    if not Jfinish[isat]:
                        Jor_data = __MakeOrbitData__(isat, J_tiempo, J_oel,
                                                     J_i1stsat)
                        for iepoc in range(2880):
                            if not np.isnan(Jor_data[iepoc, 0]):
                                Jorbit[iepoc, isat, :] = Jor_data[iepoc, :]
                                Jorbit_bool[iepoc, isat] = True

                    if np.all(Jorbit_bool[:, isat]):
                        Jfinish[isat] = True

    print("Satellite orbit data acquisition completed.", time.time() - start)

    for sta in stations:

        path_o = "{dr}/tec/{c}/{y4:04d}/{d:03d}/{s}{d:03d}0.{y2:02}o".format(
            dr=drive, c=country, y4=year4, d=day, s=sta, y2=year2)

        sat_list, Lstec, Pstec, Lstec_bool, Pstec_bool, p_o = __ReadObsFile302__(
            path_o, setfile)

        ## ここから 12/09
        # L_data_set[i,j]...number of set, sat i, epoch j
        # L_data_len[i] ... length of set i
        # L_data_sat[i] ... satelite of set i
        # L_data_begin[i] ... first epoch of set i
        L_data_set = np.full((Msat, 2880), np.nan, dtype=int)
        L_data_len = np.full((2000), 0, dtype=int)
        L_data_sat_idx = np.full((2000), 0, dtype=int)
        L_data_begin = np.full((2000), 0, dtype=int)
        # 0 ... データなし
        # 1 ... L,P, not start
        # 2 ... L,P, start
        # 3 ... L, not start
        # 4 ... L, start
        L_info = np.full((Msat, 2880), 0, dtype=int)
        # 0 ... データ無効
        # 1 ... データ有効 but 不定整数の計算にはいれない
        # 2 ... データ有効 and 不定整数の計算にいれる
        L_valid = np.full((Msat, 2880), 0, dtype=int)
        Zs = np.full((Msat, 2880), 0.0, dtype=float)
        n_data = -1

        # 各データをL,Pがあるかないかで分類
        for isat in range(len(sat_list)):
            sat_id = sat_list[isat]
            rec = XYZ(p_o[0], p_o[1], p_o[2])
            for jepoc in range(2880):
                if "G" in sat_id:
                    if not Gorbit_bool[jepoc, int(sat_id[1:3])]:
                        L_info[isat, jepoc] = 0
                        continue
                    sat = XYZ(Gorbit[jepoc, int(sat_id[1:3]),
                                     0], Gorbit[jepoc,
                                                int(sat_id[1:3]), 1],
                              Gorbit[jepoc, int(sat_id[1:3]), 2])
                if "R" in sat_id:
                    if not Rorbit_bool[jepoc, int(sat_id[1:3])]:
                        L_info[isat, jepoc] = 0
                        continue
                    sat = XYZ(Rorbit[jepoc, int(sat_id[1:3]),
                                     0], Rorbit[jepoc,
                                                int(sat_id[1:3]), 1],
                              Rorbit[jepoc, int(sat_id[1:3]), 2])
                if "E" in sat_id:
                    if not Eorbit_bool[jepoc, int(sat_id[1:3])]:
                        L_info[isat, jepoc] = 0
                        continue
                    sat = XYZ(Eorbit[jepoc, int(sat_id[1:3]),
                                     0], Eorbit[jepoc,
                                                int(sat_id[1:3]), 1],
                              Eorbit[jepoc, int(sat_id[1:3]), 2])
                if "J" in sat_id:
                    if not Jorbit_bool[jepoc, int(sat_id[1:3])]:
                        L_info[isat, jepoc] = 0
                        continue
                    sat = XYZ(Jorbit[jepoc, int(sat_id[1:3]),
                                     0], Jorbit[jepoc,
                                                int(sat_id[1:3]), 1],
                              Jorbit[jepoc, int(sat_id[1:3]), 2])
                iZ = sat_zenith(rec, sat)
                Zs[isat, jepoc] = math.degrees(iZ)
                if jepoc == 0:
                    # L,P両方データがある -> 新しいブロックの始まり
                    if Lstec_bool[isat, jepoc] and Pstec_bool[isat, jepoc]:
                        if Zenith_Threshold - 90.0 < Zs[
                                isat, jepoc] < 90.0 - Zenith_Threshold:
                            L_info[isat, jepoc] = 2
                    # Lのみある -> 4
                    elif Lstec_bool[isat, jepoc]:
                        if Zenith_Threshold - 90.0 < Zs[
                                isat, jepoc] < 90.0 - Zenith_Threshold:
                            L_info[isat, jepoc] = 4
                    # 少なくともPはない -> データなし
                    else:
                        L_info[isat, jepoc] = 0
                else:
                    now_Lstec = Lstec[isat, jepoc]
                    # ep = jepocのP,Lデータ両方あり
                    if Lstec_bool[isat, jepoc] and Pstec_bool[isat, jepoc]:
                        ibefore = -1
                        for jbefore in range(
                                1, min(Null_Threshold + 2, jepoc + 1)):
                            if L_info[isat, jepoc - jbefore] > 0:
                                ibefore = jbefore
                                break
                        # Null_Threshold以内に有効なデータがない -> start
                        if ibefore == -1:
                            if Zenith_Threshold - 90.0 < Zs[
                                    isat, jepoc] < 90.0 - Zenith_Threshold:
                                L_info[isat, jepoc] = 2
                        else:
                            before_Lstec = Lstec[isat, jepoc - ibefore]
                            # サイクルスリップあり -> start
                            if abs(now_Lstec - before_Lstec) > TEC_Threshold:
                                if Zenith_Threshold - 90.0 < Zs[
                                        isat, jepoc] < 90.0 - Zenith_Threshold:
                                    L_info[isat, jepoc] = 2
                            # サイクルスリップなし -> not start
                            else:
                                if Zenith_Threshold - 90.0 < Zs[
                                        isat, jepoc] < 90.0 - Zenith_Threshold:
                                    L_info[isat, jepoc] = 1
                    # ep = jepocはLデータのみ
                    elif Lstec_bool[isat, jepoc]:
                        ibefore = -1
                        for jbefore in range(
                                1, min(Null_Threshold + 2, jepoc + 1)):
                            if L_info[isat, jepoc - jbefore] > 0:
                                ibefore = jbefore
                        # 前に有効なデータがない -> start
                        if ibefore == -1:
                            if Zenith_Threshold - 90.0 < Zs[
                                    isat, jepoc] < 90.0 - Zenith_Threshold:
                                L_info[isat, jepoc] = 4
                        # 前に有効なデータがある
                        else:
                            before_Lstec = Lstec[isat, jepoc - ibefore]
                            # サイクルスリップあり -> start
                            if abs(now_Lstec - before_Lstec) > TEC_Threshold:
                                if Zenith_Threshold - 90.0 < Zs[
                                        isat, jepoc] < 90.0 - Zenith_Threshold:
                                    L_info[isat, jepoc] = 4
                            # サイクルスリップなし -> not start
                            else:
                                if Zenith_Threshold - 90.0 < Zs[
                                        isat, jepoc] < 90.0 - Zenith_Threshold:
                                    L_info[isat, jepoc] = 3
                    else:
                        L_info[isat, jepoc] = 0

        # どのデータを有効にするか判定
        for isat in range(Msat):
            # 前のサイクルスリップしたあとのエポック
            before_slip = 0
            # このサイクルの有効なデータ数
            count = 0
            for jepoc in range(2880):
                # データがない場合、データ数は増やさない
                if L_info[isat, jepoc] == 0:
                    pass
                # LP両方あり、スタートでない
                # データを1つ増やす
                if L_info[isat, jepoc] == 1:
                    count += 1
                # スタート
                elif L_info[isat, jepoc] == 2 or L_info[isat, jepoc] == 4:
                    # 前のサイクルスリップからのデータ数が規定値以上 -> 有効
                    if count >= Valid_Data_Length:
                        for kepoc in range(before_slip, jepoc):
                            # LP両方あるデータに関しては、不定整数の決定に使う
                            if L_info[isat, kepoc] == 1 or L_info[isat,
                                                                  kepoc] == 2:
                                L_valid[isat, kepoc] = 2
                            # Lしかない場合に関しては、不定整数の決定には使えない
                            elif L_info[isat,
                                        kepoc] == 3 or L_info[isat,
                                                              kepoc] == 4:
                                L_valid[isat, kepoc] = 1
                    before_slip = jepoc
                    count = 0
                if jepoc == 2879:
                    if count >= Valid_Data_Length:
                        for kepoc in range(before_slip, 2880):
                            # LP両方あるデータに関しては、不定整数の決定に使う
                            if L_info[isat, kepoc] == 1 or L_info[isat,
                                                                  kepoc] == 2:
                                L_valid[isat, kepoc] = 2
                            # Lしかない場合に関しては、不定整数の決定には使えない
                            elif L_info[isat,
                                        kepoc] == 3 or L_info[isat,
                                                              kepoc] == 4:
                                L_valid[isat, kepoc] = 1

        # for isat in range(Msat):
        #     for jepoc in range(2880):
        #         print(isat, jepoc, ":", Pstec[isat, jepoc], Lstec[isat, jepoc],
        #               Zs[isat, jepoc], L_info[isat, jepoc], L_valid[isat,
        #                                                             jepoc])
        #     input()

        # 各一連データの番号を割り振る
        for isat in range(Msat):
            for jepoc in range(2880):
                # データ無効
                if L_info[isat, jepoc] == 0:
                    pass
                # スタートでない
                elif L_info[isat, jepoc] == 1 or L_info[isat, jepoc] == 3:
                    # データ有効
                    if L_valid[isat, jepoc] > 0:
                        L_data_set[isat, jepoc] = n_data
                        L_data_len[n_data] += 1
                # スタート
                elif L_info[isat, jepoc] == 2 or L_info[isat, jepoc] == 4:
                    # データ有効
                    if L_valid[isat, jepoc] > 0:
                        n_data += 1
                        L_data_set[isat, jepoc] = n_data
                        L_data_sat_idx[n_data] = isat
                        L_data_begin[n_data] = jepoc
                        L_data_len[n_data] += 1

        # 12/11　ここまで
        n_data += 1
        P_O = XYZ(p_o[0], p_o[1], p_o[2])
        B_data = np.full((n_data), np.nan, dtype=float)
        for idata in range(n_data):
            isat_idx = L_data_sat_idx[idata]
            iblock_begin = L_data_begin[idata]
            # 最後のブロックの場合
            if idata == n_data - 1:
                nextblock_bgn = 2880
            # 同じ衛星に次のブロックもある場合
            # 次のブロックの始まりまで
            elif isat_idx == L_data_sat_idx[idata + 1]:
                nextblock_bgn = L_data_begin[idata + 1]
            # 次のブロックは別の衛星の場合
            else:
                nextblock_bgn = 2880

            # 分母
            lower = 0.0
            # 分子
            upper = 0.0
            if "G" in sat_list[isat_idx]:
                # 衛星のPRN番号
                isat = int(sat_list[isat_idx][1:3])
                for jepoc in range(iblock_begin, nextblock_bgn):
                    if L_valid[isat_idx, jepoc] == 2:
                        jORBIT = XYZ(Gorbit[jepoc, isat, 0],
                                     Gorbit[jepoc, isat, 1], Gorbit[jepoc,
                                                                    isat, 2])
                        jL = Lstec[isat_idx, jepoc]
                        jP = Pstec[isat_idx, jepoc]
                        t, j_ipp = specify_H(P_O, jORBIT, H_IPP * 1.0e3)
                        jZ = ipp_zenith(P_O, j_ipp, H=H_IPP * 1.0e3)
                        lower += np.power(np.sin(jZ), 2.0)
                        upper += (jP - jL) * np.power(np.sin(jZ), 2.0)
                    else:
                        pass

            elif "R" in sat_list[isat_idx]:
                # 衛星のPRN番号
                isat = int(sat_list[isat_idx][1:3])
                for jepoc in range(iblock_begin, nextblock_bgn):
                    if L_valid[isat_idx, jepoc] == 2:
                        jORBIT = XYZ(Rorbit[jepoc, isat, 0],
                                     Rorbit[jepoc, isat, 1], Rorbit[jepoc,
                                                                    isat, 2])
                        jL = Lstec[isat_idx, jepoc]
                        jP = Pstec[isat_idx, jepoc]
                        t, j_ipp = specify_H(P_O, jORBIT, H_IPP * 1.0e3)
                        jZ = ipp_zenith(P_O, j_ipp, H=H_IPP * 1.0e3)
                        lower += np.power(np.sin(jZ), 2.0)
                        upper += (jP - jL) * np.power(np.sin(jZ), 2.0)
                    else:
                        pass

            elif "E" in sat_list[isat_idx]:
                # 衛星のPRN番号
                isat = int(sat_list[isat_idx][1:3])
                for jepoc in range(iblock_begin, nextblock_bgn):
                    if L_valid[isat_idx, jepoc] == 2:
                        jORBIT = XYZ(Eorbit[jepoc, isat, 0],
                                     Eorbit[jepoc, isat, 1], Eorbit[jepoc,
                                                                    isat, 2])
                        jL = Lstec[isat_idx, jepoc]
                        jP = Pstec[isat_idx, jepoc]
                        t, j_ipp = specify_H(P_O, jORBIT, H_IPP * 1.0e3)
                        jZ = ipp_zenith(P_O, j_ipp, H=H_IPP * 1.0e3)
                        lower += np.power(np.sin(jZ), 2.0)
                        upper += (jP - jL) * np.power(np.sin(jZ), 2.0)
                    else:
                        pass
            elif "J" in sat_list[isat_idx]:
                # 衛星のPRN番号
                isat = int(sat_list[isat_idx][1:3])
                for jepoc in range(iblock_begin, nextblock_bgn):
                    if L_valid[isat_idx, jepoc] == 2:
                        jORBIT = XYZ(Jorbit[jepoc, isat, 0],
                                     Jorbit[jepoc, isat, 1], Jorbit[jepoc,
                                                                    isat, 2])
                        jL = Lstec[isat_idx, jepoc]
                        jP = Pstec[isat_idx, jepoc]
                        t, j_ipp = specify_H(P_O, jORBIT, H_IPP * 1.0e3)
                        jZ = ipp_zenith(P_O, j_ipp, H=H_IPP * 1.0e3)
                        lower += np.power(np.sin(jZ), 2.0)
                        upper += (jP - jL) * np.power(np.sin(jZ), 2.0)
                    else:
                        pass
            B = upper / lower
            B_data[idata] = B

        # バイアスがないSTECデータを計算する
        mdf_Lstec = np.full((Msat, 2880), np.nan, dtype=float)

        for idata in range(n_data):
            isat = L_data_sat_idx[idata]
            iblock_begin = L_data_begin[idata]
            if idata == n_data - 1:
                nextblock_bgn = 2880
            elif isat == L_data_sat_idx[idata + 1]:
                nextblock_bgn = L_data_begin[idata + 1]
            else:
                nextblock_bgn = 2880
            iB = B_data[idata]
            for jepoc in range(iblock_begin, nextblock_bgn):
                # データがあるときのみ
                if L_valid[isat, jepoc] > 0:
                    mdf_Lstec[isat, jepoc] = Lstec[isat, jepoc] + iB

        for isat in range(len(sat_list)):
            path_nw = "{dr}/mdf/{c}/{y4:04d}/{d:03d}/{sat}".format(
                dr=drive, c=country, y4=year4, d=day, sat=sat_list[isat])
            path_nd = path_nw + "/{sta}.mdf4".format(sta=sta)

            os.makedirs(path_nw, exist_ok=True)
            if "G" in sat_list[isat]:
                __MakeMDFFile__(path_nd,
                                mdf_Lstec[isat],
                                Lstec[isat],
                                Pstec[isat],
                                L_valid[isat, :],
                                P_O,
                                sat_list[isat],
                                Gorbit[:, int(sat_list[isat][1:3]), :],
                                Zs[isat],
                                G1_freq,
                                G2_freq,
                                ver="4.0")
            elif "R" in sat_list[isat]:
                R1_freq = R1_basefreq + GLONASS_SLOT[
                    sat_list[isat]] * R1_deltafreq
                R2_freq = R2_basefreq + GLONASS_SLOT[
                    sat_list[isat]] * R2_deltafreq
                __MakeMDFFile__(path_nd,
                                mdf_Lstec[isat],
                                Lstec[isat],
                                Pstec[isat],
                                L_valid[isat, :],
                                P_O,
                                sat_list[isat],
                                Rorbit[:, int(sat_list[isat][1:3]), :],
                                Zs[isat],
                                R1_freq,
                                R2_freq,
                                ver="4.0")
            elif "E" in sat_list[isat]:
                __MakeMDFFile__(path_nd,
                                mdf_Lstec[isat],
                                Lstec[isat],
                                Pstec[isat],
                                L_valid[isat, :],
                                P_O,
                                sat_list[isat],
                                Eorbit[:, int(sat_list[isat][1:3]), :],
                                Zs[isat],
                                E1_freq,
                                E2_freq,
                                ver="4.0")
            elif "J" in sat_list[isat]:
                __MakeMDFFile__(path_nd,
                                mdf_Lstec[isat],
                                Lstec[isat],
                                Pstec[isat],
                                L_valid[isat, :],
                                P_O,
                                sat_list[isat],
                                Jorbit[:, int(sat_list[isat][1:3]), :],
                                Zs[isat],
                                J1_freq,
                                J2_freq,
                                ver="4.0")
        print(sta, "end ", time.time() - start)
    return


def __RNX2MDF212__(drive: str, year4: int, day: int, stations: list,
                   setfile: str):
    return


def __RNX2MDF210__(drive: str, year4: int, day: int, stations: list,
                   setfile: str):
    return


def __RNX2MDF__(
    drive: str,
    country: str,
    year4: int,
    day: int,
    stations: list,
    setfile: str,
    ver: int,
):
    if ver == 302:
        return __RNX2MDF302__(drive, country, year4, day, stations, setfile)
    elif ver == 212:
        return __RNX2MDF212__(drive, country, year4, day, stations, setfile)
    elif ver == 210:
        return __RNX2MDF210__(drive, country, year4, day, stations, setfile)
    else:
        raise ValueError("Select ver from 302, 212, 210")
