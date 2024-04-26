import os
import glob
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, spsolve, svds
from scipy.optimize import lsq_linear, Bounds
from tqdm import tqdm
import datetime

from f_class import XYZ, BLH, STECRecord
from function import specify_H

STEP = 0
H_IPP = 0.0
H_IPPsup = 0.0
H_IPPinf = 0.0
mLAT = 0.0
MLAT = 0.0
mLON = 0.0
MLON = 0.0
nLAT = 0
nLON = 0
a1b = np.full((0), 0.0, dtype=float)
a1l = np.full((0), 0.0, dtype=float)
a1t = np.full((0), 0, dtype=float)
nTIME = 0

BIAS_CODE = {"GPS": -1, "GLONASS": -1, "GALILEO": -1, "QZSS": -1}


def __setting__(file):
    global STEP, H_IPP, H_IPPinf, H_IPPsup
    global mLAT, MLAT, mLON, MLON
    global nLAT, nLON, nTIME
    global a1b, a1l, a1t
    global BIAS_CODE
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "STEP FOR MDF2BIAS" in line:
                STEP = int(line.split()[0])
                a1t = np.arange(0, 2880, STEP)
                nTIME = a1t.shape[0]
            if "IPP HEIGHT" in line:
                H_IPP = float(line.split()[0])
            if "IPP inf HEIGHT" in line:
                H_IPPinf = float(line.split()[0])
            if "IPP sup HEIGHT" in line:
                H_IPPsup = float(line.split()[0])
            if "mLAT" in line:
                mLAT = float(line.split()[0])
            if "MLAT" in line:
                MLAT = float(line.split()[0])
            if "mLON" in line:
                mLON = float(line.split()[0])
            if "MLON" in line:
                MLON = float(line.split()[0])

            if "NLAT FOR MDF2BIAS" in line:
                idx = lines.index(line)
                nLAT = int(line.split()[0])
                a1b = np.full((nLAT + 1), 0.0, dtype=float)
                for ilat in range(nLAT + 1):
                    jlat = ilat % 10
                    if jlat == 0:
                        idx += 1
                    a1b[ilat] = float(lines[idx].split()[jlat])
                idx += 1
                nLON = int(lines[idx].split()[0])
                a1l = np.full((nLON + 1), 0.0, dtype=float)
                for ilon in range(nLON + 1):
                    jlon = ilon % 10
                    if jlon == 0:
                        idx += 1
                    a1l[ilon] = float(lines[idx].split()[jlon])

            if "GPS BIAS CODE" in line:
                BIAS_CODE["GPS"] = int(line.split()[0])
            if "GLONASS BIAS CODE" in line:
                BIAS_CODE["GLONASS"] = int(line.split()[0])
            if "GALILEO BIAS CODE" in line:
                BIAS_CODE["GALILEO"] = int(line.split()[0])
            if "QZSS BIAS CODE" in line:
                BIAS_CODE["QZSS"] = int(line.split()[0])


def __MDF2BIAS__(drive: str, country: str, year4: int, day: int, setting: str,
                 ver: str):

    __setting__(setting)

    global BIAS_CODE
    global a1b, a1l, a1t
    global nLAT, nLON, nTIME
    global H_IPP, H_IPPinf, H_IPPsup

    sat_folders = sorted(
        glob.glob(f"{drive}/mdf/{country}/{year4:04d}/{day:03d}/*"))

    records = []
    receivers = []
    satellites = []
    for it in range(nTIME):
        records.append([])

    nRECORD = 0

    for sat_folder in tqdm(sat_folders):
        if os.path.isdir(sat_folder):
            sat_id = sat_folder[-3:]
            if not sat_id in satellites:
                satellites.append(sat_id)
            mdf_files = sorted(glob.glob(sat_folder + "/*.mdf4"))
            for mdf_file in mdf_files:
                rec_id = mdf_file[-9:-5]
                if not rec_id in receivers:
                    receivers.append(rec_id)
                with open(mdf_file, "r") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        if "END OF HEADER" in line:
                            break
                        if "Receiver Position" in line:
                            X = float(line.split()[4])
                            Y = float(line.split()[5])
                            Z = float(line.split()[6])
                            rec = XYZ(X, Y, Z)

                    while True:
                        line = f.readline()
                        if not line:
                            break
                        time = float(line.split()[0])
                        epoch = round(time * 120)
                        if epoch in a1t:
                            stec = float(line.split()[1])
                            sX = float(line.split()[4])
                            sY = float(line.split()[5])
                            sZ = float(line.split()[6])
                            sat = XYZ(sX, sY, sZ)
                            record = STECRecord(stec, sat, rec, sat_id, rec_id)
                            records[np.where(
                                a1t == epoch)[0][0]].append(record)
                            nRECORD += 1

    nREC = len(receivers)
    nSAT = len(satellites)
    nBIAS = max(BIAS_CODE.values()) + 1
    nBOX = nLAT * nLON * nTIME

    ncol = nRECORD
    nrow = nLAT * nLON * nTIME + nSAT + nREC * nBIAS
    print(ncol, nrow)

    A = lil_matrix((ncol, nrow))
    b = np.full((ncol, 1), 0.0, dtype=float)

    icol = 0
    for iepoch in tqdm(range(nTIME)):
        tmp_records = records[iepoch]
        for jrecord in range(len(tmp_records)):
            record: STECRecord = tmp_records[jrecord]
            sat_pos = record.sat
            rec_pos = record.rec
            tipp, ipp = specify_H(rec_pos, sat_pos, H_IPP * 1.0e3)
            tipp, ippinf = specify_H(rec_pos, sat_pos, H_IPPinf * 1.0e3)
            tipp, ippsup = specify_H(rec_pos, sat_pos, H_IPPsup * 1.0e3)
            S1 = (ippinf - ippsup).L2()
            S0 = (H_IPPsup - H_IPPinf) * 1.0e3
            ipp_b = ipp.to_BLH().b
            ipp_l = ipp.to_BLH().l
            ipp_bidx = np.searchsorted(a1b, ipp_b) - 1
            ipp_lidx = np.searchsorted(a1l, ipp_l) - 1
            isat = satellites.index(record.sat_id)
            irec = receivers.index(record.rec_id)
            if "G" in record.sat_id:
                ibias = BIAS_CODE["GPS"]
            if "R" in record.sat_id:
                ibias = BIAS_CODE["GLONASS"]
            if "E" in record.sat_id:
                ibias = BIAS_CODE["GALILEO"]
            if "J" in record.sat_id:
                ibias = BIAS_CODE["QZSS"]
            if ibias > -1:
                b[icol, 0] = record.stec
                A[icol, nBOX + isat] = 1.0
                A[icol, nBOX + nSAT + nREC * ibias + irec] = 1.0  # 01/09
                A[icol,
                  nLAT * nLON * iepoch + ipp_bidx * nLON + ipp_lidx] = S1 / S0
                icol += 1
    A = csr_matrix(A)

    # Select Solver

    # 1. lsqr
    X, istop, itn, r1norm = lsqr(A, b, atol=1.0e-7)[:4]

    # 2. spsolve
    # X = spsolve(A, b)
    # r1norm = np.linalg.norm(A @ X - b, ord=2)

    # 3. svds
    # U, S, VT = svds(A, k=1000, tol=1.0e-7)
    # print(S)
    # A_inv = VT.T @ np.diag(1 / S) @ U.T
    # X = A_inv @ b
    # r1norm = np.linalg.norm(A @ X - b, ord=2)

    # 4. lsq_linear
    # lb = np.full((nrow), 0.0, dtype=float)
    # for i in range(nBOX, nrow):
    #     lb[i] = -np.inf
    # ub = np.full((nrow), np.inf, dtype=float)
    # bound = Bounds(lb, ub)
    # result = lsq_linear(A, b, bounds=bound, tol=1.0e-8)
    # X = result.x
    # r1norm = result.cost

    os.makedirs(f"{drive}/vtecmap/{country}/{year4:04d}/{day:03d}/",
                exist_ok=True)
    biasfolder = f"{drive}/vtecmap/{country}/{year4:04d}/{day:03d}/{ver}.vtecmap"
    with open(biasfolder, "w") as f:
        print(f"# RNX2VTX VERSION : {ver}", file=f)
        print(f"# UTICTIME : {datetime.datetime.now()}", file=f)
        print("#", file=f)
        print(f"# NUMBER OF TIME : {nTIME:04d}", file=f)
        for itime in range(nTIME):
            if itime % 10 == 9:
                print(f"{a1t[itime]:04d} ", file=f)
            else:
                print(f"{a1t[itime]:04d} ", end="", file=f)
        print("", file=f)
        print(f"# NUMBER OF LATITUDE : {nLAT:03d}", file=f)
        for ilat in range(nLAT):
            if ilat % 10 == 9:
                print(f"{a1b[ilat]:05.1f} ", file=f)
            else:
                print(f"{a1b[ilat]:05.1f} ", end="", file=f)
        print("", file=f)
        print(f"# NUMBER OF LONGITUDE : {nLON:03d}", file=f)
        for ilon in range(nLON):
            if ilon % 10 == 9:
                print(f"{a1l[ilon]:05.1f} ", file=f)
            else:
                print(f"{a1l[ilon]:05.1f} ", end="", file=f)
        print("", file=f)
        print("#", file=f)
        print(f"# IPP HEIGHT [km] : {H_IPP:05.1f}", file=f)
        print(f"# IPP inf HEIGHT [km] : {H_IPPinf:05.1f}", file=f)
        print(f"# IPP sup HEIGHT [km] : {H_IPPsup:05.1f}", file=f)
        print("#", file=f)
        print(f"# NUMBER OF SATELLITE : {nSAT}", file=f)
        print(f"# NUMBER OF RECEIVER : {nREC}", file=f)
        print("#", file=f)
        print(f"# SETTING : {setting}", file=f)
        print("#", file=f)
        print(f"# NUMBER OF FORMULA : {ncol}", file=f)
        print(f"# RESIDUAL NORM OF b-Ax : {r1norm}", file=f)
        print("#", file=f)
        print("# BIAS CODE", file=f)
        for k, v in BIAS_CODE.items():
            print(f"# {k} {v}", file=f)
        print("#", file=f)
        print("# END OF HEADER", file=f)

        print("", file=f)
        print("## VTECMAP", file=f)
        for itime in range(nTIME):
            for jlat in range(nLAT):
                for klon in range(nLON):
                    print(f"{X[nLAT*nLON*itime+nLON*jlat+klon]:07.3f}",
                          end=" ",
                          file=f)
                print("", file=f)
            print("", file=f)

        print("", file=f)
        print("## SATELLITE BIAS", file=f)
        for isat in range(nSAT):
            print(f"{satellites[isat]} {X[nBOX+isat]:08.3f}", file=f)

        print("", file=f)
        print("## RECEIVER BIAS", file=f)
        for irec in range(nREC):
            print(f"{receivers[irec]}", end=" ", file=f)
            for jbias in range(nBIAS):
                print(f"{X[nBOX+nSAT+nREC*jbias+irec]:08.3f}", end=" ", file=f)
            print("", file=f)

    return
