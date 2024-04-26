from f_class import EXPERIMENT, INPUT, XYZ, SETTING
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def ReadTomoiF(drive, exp: EXPERIMENT) -> INPUT:
    country = exp.c
    year4 = exp.y
    day = exp.d
    ep = exp.ep

    Tomography_input = "{dr}/tomoiF/{c}/{y:04d}/{d:03d}/{ep:04d}.tomoif".format(
        dr=drive, c=country, y=year4, d=day, ep=ep)

    _input = INPUT()

    with open(Tomography_input, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                tec = float(line.split()[0])
                sat = XYZ(float(line.split()[1]), float(line.split()[2]),
                          float(line.split()[3]))
                rec = XYZ(float(line.split()[4]), float(line.split()[5]),
                          float(line.split()[6]))
                sat_id = line.split()[7]
                rec_id = line.split()[8]
                _input.__addF__(tec, sat, rec, sat_id, rec_id)

    return _input


def Read_A(drive, exp: EXPERIMENT) -> csr_matrix:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep
    tomoc = "{dr}/tomoc/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomoc".format(
        dr=drive, c=country, y=year4, d=day, cd=code, ep=ep)
    with open(tomoc, "r") as f:
        line = f.readline()
        L = int(line.split()[0])
        M = int(line.split()[1])
        A = lil_matrix((L, M))
        while True:
            line = f.readline()
            if not line or len(line.split()) < 3:
                break
            else:
                i = int(line.split()[0])
                j = int(line.split()[1])
                l = float(line.split()[2])
                A[i, j] = l
    return csr_matrix(A)


def Read_B(drive, exp: EXPERIMENT) -> np.ndarray:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep
    tomob = "{dr}/tomob/{c}/{y:04d}/{d:03d}/{ep:04d}.tomob".format(dr=drive,
                                                                   c=country,
                                                                   y=year4,
                                                                   d=day,
                                                                   ep=ep)

    with open(tomob, "r") as f:
        line = f.readline()
        L = int(line.split()[0])
        B = np.full((L, 1), 0.0, dtype=float)
        while True:
            line = f.readline()
            if not line:
                break
            i = int(line.split()[0])
            b = float(line.split()[1])
            B[i, 0] = b

    return B


def Read_X(drive, exp: EXPERIMENT) -> np.ndarray:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep
    tomo = "{dr}/tomo/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomo".format(dr=drive,
                                                                     c=country,
                                                                     y=year4,
                                                                     d=day,
                                                                     cd=code,
                                                                     ep=ep)
    with open(tomo, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                break
            if "Number of Boxel (All area)" in line:
                line = f.readline()
                n_b = int(line.split()[3])
                line = f.readline()
                n_l = int(line.split()[3])
                line = f.readline()
                n_h = int(line.split()[3])

        n_all = n_b * n_l * n_h
        datas = np.full((n_all, 1), 0.0, dtype=float)
        for kh in range(n_h):
            line = f.readline()
            for jb in range(n_b):
                line = f.readline()
                for il in range(n_l):
                    datas[kh * n_b * n_l + jb * n_l + il,
                          0] = float(line.split()[il])
    return datas


def Read_Setting(file: str) -> SETTING:
    a1h = np.full((0), 0.0, dtype=float)
    a1b = np.full((0), 0.0, dtype=float)
    a1l = np.full((0), 0.0, dtype=float)
    cof = np.full((0), 0.0, dtype=float)
    lam = 0.0
    alpha = 0.0
    with open(file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "# A1H" in line:
                n_h = int(line[6:9])
                a1h = np.full((n_h), 0.0, dtype=float)
                for ih in range(n_h):
                    jh = ih % 5
                    if jh == 0:
                        line = f.readline()
                    if not line:
                        break
                    a1h[ih] = float(line[8 * jh:7 + 8 * jh])
            if "# A1B" in line:
                n_b = int(line[6:9])
                a1b = np.full((n_b), 0.0, dtype=float)
                for ib in range(n_b):
                    jb = ib % 5
                    if jb == 0:
                        line = f.readline()
                    if not line:
                        break
                    a1b[ib] = float(line[8 * jb:7 + 8 * jb])
            if "# A1L" in line:
                n_l = int(line[6:9])
                a1l = np.full((n_l), 0.0, dtype=float)
                for il in range(n_l):
                    jl = il % 5
                    if jl == 0:
                        line = f.readline()
                    if not line:
                        break
                    a1l[il] = float(line[8 * jl:7 + 8 * jl])
            if "# COEFF" in line:
                n_cof = int(line[8:11])
                cof = np.full((n_cof), 0.0, dtype=float)
                for ic in range(n_cof):
                    jc = ic % 4
                    if jc == 0:
                        line = f.readline()
                    if not line:
                        break
                    cof[ic] = float(line[15 * jc:14 + 15 * jc])
            if "# LAMBDA" in line:
                lam = float(line[9:14])
            if "# ALPHA" in line:
                alpha = float(line[8:11])

    setting = SETTING(a1h, a1b, a1l, lam, cof, alpha)

    return setting
