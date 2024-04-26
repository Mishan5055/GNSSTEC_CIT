import numpy as np
import datetime
import math

Mrecord = 0
Msat = 0


def __setfile__(file):
    global Mrecord, Msat
    with open(file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "MAX RECORD" in line:
                Mrecord = int(line.split()[0])
            if "MAX SATELLITE" in line:
                Msat = int(line.split()[0])


def __ReadNavFile302__(path_n, obs_day, setfile):
    __setfile__(setfile)
    # 航法データを格納しておく変数
    oel = np.full((Mrecord, 28), np.nan, dtype=float)
    # データの衛星番号を保持しておく用 (衛星系ごとに別。G01なら01)
    idsat = np.zeros(Mrecord, dtype=int)
    # 基準となる時点(obs_day 00:00:00UT)からの経過時間(hour)
    tiempo = np.zeros(Mrecord, dtype=float)
    # 衛星ごとに格納
    i1stsat = [[] for i in range(Msat)]

    with open(path_n, "r") as f:
        lines = [s.rstrip() for s in f.readlines()]

    L = len(lines)
    idx = 0
    krec = 0
    header_flag = True

    while idx < L:
        if header_flag:
            idx += 1
        if "END OF HEADER" in lines[idx]:
            header_flag = False
            idx += 1
            if not lines[idx]:
                break
            continue

        if not header_flag:
            idsat[krec] = int(lines[idx][1:3])
            iy = int(lines[idx][4:8])
            imonth = int(lines[idx][9:11])
            iday = int(lines[idx][12:14])
            ih = int(lines[idx][15:17])
            imin = int(lines[idx][18:20])
            isec = int(lines[idx][21:23])

            dt2 = datetime.datetime(year=iy,
                                    month=imonth,
                                    day=iday,
                                    hour=ih,
                                    minute=imin,
                                    second=isec)
            td = dt2 - obs_day
            tiempo[krec] = td.total_seconds() / 3600.0
            if not lines[idx + 1]:
                break

            for irec in range(7):
                ii = irec * 4
                for j in range(4):
                    try:
                        m_str = lines[idx + irec + 1][4 + 19 * j:19 +
                                                      19 * j].strip()
                        e_str = lines[idx + irec + 1][20 + 19 * j:23 +
                                                      19 * j].strip()
                    except:
                        idsat[krec] = 0
                        break
                    if not m_str == "" and not e_str == "":
                        oel[krec][ii +
                                  j] = float(m_str) * math.pow(10, int(e_str))
                    else:
                        oel[krec][ii + j] = np.nan
            idx += 8
            if idx < L and not lines[idx]:
                break
            krec += 1

    for k in range(krec):
        i1stsat[idsat[k]].append(k)

    return tiempo, oel, i1stsat


def __ReadGLONASSNavFile302__(path_g, obs_day: datetime.datetime, setfile):
    __setfile__(setfile)

    # 航法データを格納しておく変数
    oel = np.full((Mrecord, 9), np.nan, dtype=float)
    # データの衛星番号を保持しておく用 (衛星系ごとに別。G01なら01)
    idsat = np.zeros(Mrecord, dtype=int)
    # 基準となる時点(obs_day 00:00:00UT)からの経過時間(hour)
    tiempo = np.zeros(Mrecord, dtype=float)
    # 衛星ごとに格納
    i1stsat = [[] for i in range(Msat)]

    krec = 0
    with open(path_g, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                break

        while True:
            line = f.readline()
            if not line:
                break
            if not line[0:3]:
                pass
            else:
                sat_id = int(line[1:3])
                idsat[krec] = sat_id
                iyear = int(line[4:8])
                imonth = int(line[9:11])
                iday = int(line[12:14])
                ihour = int(line[15:17])
                iminute = int(line[18:20])
                isecond = int(line[21:23])

                dt2 = datetime.datetime(year=iyear,
                                        month=imonth,
                                        day=iday,
                                        hour=ihour,
                                        minute=iminute,
                                        second=isecond)

                td = dt2 - obs_day
                tiempo[krec] = td.total_seconds() / 3600.0

                for ioel in range(3):
                    line = f.readline()
                    if not line:
                        break
                    for joel in range(3):
                        d_str = line[4 + 19 * joel:19 + 19 * joel]
                        e_str = line[20 + 19 * joel:23 + 19 * joel]
                        if d_str and e_str:
                            oel[krec,
                                ioel * 3 + joel] = float(d_str) * math.pow(
                                    10,
                                    float(e_str) + 3)

                krec += 1

    for k in range(krec):
        i1stsat[idsat[k]].append(k)

    return tiempo, oel, i1stsat
