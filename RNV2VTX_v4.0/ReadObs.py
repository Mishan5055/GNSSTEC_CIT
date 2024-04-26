import numpy as np
import os
import datetime
import math

Msat = 0

vel_light = 2.9979 * math.pow(10, 8)
K = 0.0

lG1_code = ""
lG2_code = ""
pG1_code = ""
pG2_code = ""

lR1_code = ""
lR2_code = ""
pR1_code = ""
pR2_code = ""

lE1_code = ""
lE2_code = ""
pE1_code = ""
pE2_code = ""

lJ1_code = ""
lJ2_code = ""
pJ1_code = ""
pJ2_code = ""

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
    global Msat, K
    global lG1_code, lG2_code, pG1_code, pG2_code
    global lR1_code, lR2_code, pR1_code, pR2_code
    global lE1_code, lE2_code, pE1_code, pE2_code
    global lJ1_code, lJ2_code, pJ1_code, pJ2_code
    global G1_freq, G2_freq
    global R1_basefreq, R2_basefreq, R1_deltafreq, R2_deltafreq
    global E1_freq, E2_freq
    global J1_freq, J2_freq
    with open(file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "MAX SATELLITE" in line:
                Msat = int(line.split()[0])

            if "TECK" in line:
                K = float(line.split()[0])

            if "f1GPS" in line:
                G1_freq = int(line.split()[0])
                pG1_code = line.split()[1]
                lG1_code = line.split()[2]
            if "f2GPS" in line:
                G2_freq = int(line.split()[0])
                pG2_code = line.split()[1]
                lG2_code = line.split()[2]
            if "f1GLONASS" in line:
                R1_basefreq = int(line.split()[0])
                R1_deltafreq = int(line.split()[1])
                pR1_code = line.split()[2]
                lR1_code = line.split()[3]
            if "f2GLONASS" in line:
                R2_basefreq = int(line.split()[0])
                R2_deltafreq = int(line.split()[1])
                pR2_code = line.split()[2]
                lR2_code = line.split()[3]
            if "f1GALILEO" in line:
                E1_freq = int(line.split()[0])
                pE1_code = line.split()[1]
                lE1_code = line.split()[2]
            if "f2GALILEO" in line:
                E2_freq = int(line.split()[0])
                pE2_code = line.split()[1]
                lE2_code = line.split()[2]
            if "f1QZSS" in line:
                J1_freq = int(line.split()[0])
                pJ1_code = line.split()[1]
                lJ1_code = line.split()[2]
            if "f2QZSS" in line:
                J2_freq = int(line.split()[0])
                pJ2_code = line.split()[1]
                lJ2_code = line.split()[2]
            if "GLONASS SLOT" in line:
                for i in range(8):
                    sat_id = line[7 * i:3 + 7 * i]
                    slot = line[4 + 7 * i:6 + 7 * i]
                    if not sat_id in GLONASS_SLOT:
                        GLONASS_SLOT[sat_id] = int(slot)


def __ReadObsFile302__(path_o, setfile):

    __setfile__(setfile)

    global lG1_code, lG2_code, pG1_code, pG2_code
    global lR1_code, lR2_code, pR1_code, pR2_code
    global lE1_code, lE2_code, pE1_code, pE2_code
    global lJ1_code, lJ2_code, pJ1_code, pJ2_code

    global G1_freq, G2_freq
    global R1_basefreq, R2_basefreq, R1_deltafreq, R2_deltafreq
    global E1_freq, E2_freq
    global J1_freq, J2_freq

    global GLONASS_SLOT

    p_o = np.zeros(3)
    with open(path_o, "r") as f:
        lG1_num = -1
        lG2_num = -1
        pG1_num = -1
        pG2_num = -1

        lR1_num = -1
        lR2_num = -1
        pR1_num = -1
        pR2_num = -1

        lE1_num = -1
        lE2_num = -1
        pE1_num = -1
        pE2_num = -1

        lJ1_num = -1
        lJ2_num = -1
        pJ1_num = -1
        pJ2_num = -1

        while True:
            line = f.readline()
            if "APPROX POSITION XYZ" in line:
                p_o[0] = float(line[1:14])
                p_o[1] = float(line[15:28])
                p_o[2] = float(line[29:42])
            if "OBS TYPES" in line:
                if line[4:6].strip() == "":
                    continue
                wave_sum = int(line[4:6])
                sat_type = line[0:1]
                if sat_type == "G":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = line[7 + 4 * j:10 + 4 * j]
                        if wave == lG1_code:
                            lG1_num = i
                        elif wave == lG2_code:
                            lG2_num = i
                        elif wave == pG1_code:
                            pG1_num = i
                        elif wave == pG2_code:
                            pG2_num = i
                        if j == 12:
                            line = f.readline()
                if sat_type == "R":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = line[7 + 4 * j:10 + 4 * j]
                        if wave == lR1_code:
                            lR1_num = i
                        elif wave == lR2_code:
                            lR2_num = i
                        elif wave == pR1_code:
                            pR1_num = i
                        elif wave == pR2_code:
                            pR2_num = i
                        if j == 12:
                            line = f.readline()
                if sat_type == "E":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = line[7 + 4 * j:10 + 4 * j]
                        if wave == lE1_code:
                            lE1_num = i
                        elif wave == lE2_code:
                            lE2_num = i
                        elif wave == pE1_code:
                            pE1_num = i
                        elif wave == pE2_code:
                            pE2_num = i
                        if j == 12:
                            line = f.readline()
                if sat_type == "J":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = line[7 + 4 * j:10 + 4 * j]
                        if wave == lJ1_code:
                            lJ1_num = i
                        elif wave == lJ2_code:
                            lJ2_num = i
                        elif wave == pJ1_code:
                            pJ1_num = i
                        elif wave == pJ2_code:
                            pJ2_num = i
                        if j == 12:
                            line = f.readline()
            if "END OF HEADER" in line:
                break
            if not line:
                break

        Pstec = np.full((Msat, 2880), np.nan, dtype=float)
        Pstec_bool = np.full((Msat, 2880), False, dtype=bool)
        Lstec = np.full((Msat, 2880), np.nan, dtype=float)
        Lstec_bool = np.full((Msat, 2880), False, dtype=bool)
        sat_list = []

        while True:
            line = f.readline()
            if not line:
                break
            iyear = int(line[2:6])
            imonth = int(line[7:9])
            iday = int(line[10:12])
            ih = int(line[13:15])
            imin = int(line[16:18])
            isec = int(line[19:21])
            sat_sum = int(line[33:35])
            iepoc = isec // 30 + 2 * imin + 120 * ih

            for irec in range(sat_sum):
                line = f.readline()
                sat_code = line[0:3]
                if not sat_code in sat_list:
                    sat_list.append(sat_code)
                idx = sat_list.index(sat_code)
                if sat_code[0:1] == "G":
                    strl1 = line[3 + 16 * lG1_num:17 + 16 * lG1_num].strip()
                    strl2 = line[3 + 16 * lG2_num:17 + 16 * lG2_num].strip()
                    strp1 = line[3 + 16 * pG1_num:17 + 16 * pG1_num].strip()
                    strp2 = line[3 + 16 * pG2_num:17 + 16 * pG2_num].strip()
                    l1_bool = False
                    l2_bool = False
                    p1_bool = False
                    p2_bool = False

                    if not strl1 == "":
                        fltl1 = float(strl1)
                        l1_bool = True
                    if not strl2 == "":
                        fltl2 = float(strl2)
                        l2_bool = True
                    if not strp1 == "":
                        fltp1 = float(strp1)
                        p1_bool = True
                    if not strp2 == "":
                        fltp2 = float(strp2)
                        p2_bool = True
                    if l1_bool and l2_bool:
                        Lstec[idx, iepoc] = (
                            2 * math.pow(G1_freq * G2_freq, 2.0) *
                            (vel_light * fltl1 / G1_freq -
                             vel_light * fltl2 / G2_freq) /
                            (1.0e16 * K *
                             (math.pow(G1_freq, 2.0) - math.pow(G2_freq, 2.0)))
                        )
                        Lstec_bool[idx, iepoc] = True
                    if p1_bool and p2_bool:
                        Pstec[idx, iepoc] = (
                            2 * math.pow(G1_freq * G2_freq, 2.0) *
                            (fltp2 - fltp1) /
                            (1.0e16 * K *
                             (math.pow(G1_freq, 2.0) - math.pow(G2_freq, 2.0)))
                        )
                        Pstec_bool[idx, iepoc] = True

                if sat_code[0:1] == "R":
                    strl1 = line[3 + 16 * lR1_num:17 + 16 * lR1_num].strip()
                    strl2 = line[3 + 16 * lR2_num:17 + 16 * lR2_num].strip()
                    strp1 = line[3 + 16 * pR1_num:17 + 16 * pR1_num].strip()
                    strp2 = line[3 + 16 * pR2_num:17 + 16 * pR2_num].strip()
                    l1_bool = False
                    l2_bool = False
                    p1_bool = False
                    p2_bool = False

                    if not strl1 == "":
                        fltl1 = float(strl1)
                        l1_bool = True
                    if not strl2 == "":
                        fltl2 = float(strl2)
                        l2_bool = True
                    if not strp1 == "":
                        fltp1 = float(strp1)
                        p1_bool = True
                    if not strp2 == "":
                        fltp2 = float(strp2)
                        p2_bool = True

                    slot = GLONASS_SLOT[sat_code]
                    R1_freq = R1_basefreq + slot * R1_deltafreq
                    R2_freq = R2_basefreq + slot * R2_deltafreq

                    if l1_bool and l2_bool:
                        Lstec[idx, iepoc] = (
                            2 * math.pow(R1_freq * R2_freq, 2.0) *
                            (vel_light * fltl1 / R1_freq -
                             vel_light * fltl2 / R2_freq) /
                            (1.0e16 * K *
                             (math.pow(R1_freq, 2.0) - math.pow(R2_freq, 2.0)))
                        )
                        Lstec_bool[idx, iepoc] = True
                    if p1_bool and p2_bool:
                        Pstec[idx, iepoc] = (
                            2 * math.pow(R1_freq * R2_freq, 2.0) *
                            (fltp2 - fltp1) /
                            (1.0e16 * K *
                             (math.pow(R1_freq, 2.0) - math.pow(R2_freq, 2.0)))
                        )
                        Pstec_bool[idx, iepoc] = True

                if sat_code[0:1] == "E":
                    strl1 = line[3 + 16 * lE1_num:17 + 16 * lE1_num].strip()
                    strl2 = line[3 + 16 * lE2_num:17 + 16 * lE2_num].strip()
                    strp1 = line[3 + 16 * pE1_num:17 + 16 * pE1_num].strip()
                    strp2 = line[3 + 16 * pE2_num:17 + 16 * pE2_num].strip()
                    l1_bool = False
                    l2_bool = False
                    p1_bool = False
                    p2_bool = False

                    if not strl1 == "":
                        fltl1 = float(strl1)
                        l1_bool = True
                    if not strl2 == "":
                        fltl2 = float(strl2)
                        l2_bool = True
                    if not strp1 == "":
                        fltp1 = float(strp1)
                        p1_bool = True
                    if not strp2 == "":
                        fltp2 = float(strp2)
                        p2_bool = True
                    if l1_bool and l2_bool:
                        Lstec[idx, iepoc] = (
                            2 * math.pow(E1_freq * E2_freq, 2.0) *
                            (vel_light * fltl1 / E1_freq -
                             vel_light * fltl2 / E2_freq) /
                            (1.0e16 * K *
                             (math.pow(E1_freq, 2.0) - math.pow(E2_freq, 2.0)))
                        )
                        Lstec_bool[idx, iepoc] = True
                    if p1_bool and p2_bool:
                        Pstec[idx, iepoc] = (
                            2 * math.pow(E1_freq * E2_freq, 2.0) *
                            (fltp2 - fltp1) /
                            (1.0e16 * K *
                             (math.pow(E1_freq, 2.0) - math.pow(E2_freq, 2.0)))
                        )
                        Pstec_bool[idx, iepoc] = True

                if sat_code[0:1] == "J":
                    strl1 = line[3 + 16 * lJ1_num:17 + 16 * lJ1_num].strip()
                    strl2 = line[3 + 16 * lJ2_num:17 + 16 * lJ2_num].strip()
                    strp1 = line[3 + 16 * pJ1_num:17 + 16 * pJ1_num].strip()
                    strp2 = line[3 + 16 * pJ2_num:17 + 16 * pJ2_num].strip()
                    l1_bool = False
                    l2_bool = False
                    p1_bool = False
                    p2_bool = False

                    if not strl1 == "":
                        fltl1 = float(strl1)
                        l1_bool = True
                    if not strl2 == "":
                        fltl2 = float(strl2)
                        l2_bool = True
                    if not strp1 == "":
                        fltp1 = float(strp1)
                        p1_bool = True
                    if not strp2 == "":
                        fltp2 = float(strp2)
                        p2_bool = True
                    if l1_bool and l2_bool:
                        Lstec[idx, iepoc] = (
                            2 * math.pow(J1_freq * J2_freq, 2.0) *
                            (vel_light * fltl1 / J1_freq -
                             vel_light * fltl2 / J2_freq) /
                            (1.0e16 * K *
                             (math.pow(J1_freq, 2.0) - math.pow(J2_freq, 2.0)))
                        )
                        Lstec_bool[idx, iepoc] = True
                    if p1_bool and p2_bool:
                        Pstec[idx, iepoc] = (
                            2 * math.pow(J1_freq * J2_freq, 2.0) *
                            (fltp2 - fltp1) /
                            (1.0e16 * K *
                             (math.pow(J1_freq, 2.0) - math.pow(J2_freq, 2.0)))
                        )
                        Pstec_bool[idx, iepoc] = True

    return sat_list, Lstec, Pstec, Lstec_bool, Pstec_bool, p_o
