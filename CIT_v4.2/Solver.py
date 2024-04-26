from f_class import EXPERIMENT, SETTING
from Reader import Read_A, Read_B, Read_X, ReadTomoiF
from Writer import Write_X, Write_B, Write_A
from Generator import Gen_A, Gen_H, Gen_Y, Gen_B, Gen_X0

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve, bicgstab
import os
import math
import time
import datetime

num_iter = 0


def __callback__(x: np.ndarray):
    global num_iter
    num_iter += 1


def __SettingProcess__(drive, exp: EXPERIMENT, _setting: SETTING):
    _input = ReadTomoiF(drive, exp)
    lil_A = Gen_A(drive, _input, _setting)
    Write_A(lil_A, drive, exp)
    B = Gen_B(exp, _input, _setting)
    Write_B(B, drive, exp)
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    epoch = exp.ep
    logfile = f"{drive}/tomo/{country}/{year4:04d}/{day:03d}/{code}/log.txt"
    with open(logfile, "a") as f:
        print(
            f"SettingProcessEnd : {country} {year4:04d} {day:03d} {code} {epoch:04d} : {datetime.datetime.now()}",
            file=f)


def __SolveFirstProblem__(drive, exp: EXPERIMENT, setting: SETTING):
    global num_iter
    num_iter = 0

    n_all = setting.n_all
    lamb = setting.lamb

    if n_all < 7000:
        A = Read_A(drive, exp)
        B = Read_B(drive, exp)
        H = setting.H
        Y = Gen_Y(exp, setting)

        sco = float(csr_matrix.trace(A.T * A) / setting.trH)
        co = math.sqrt(lamb * sco)
        X = spsolve(A.T * A + co * co * H.T * H, A.T * B + co * co * H.T * Y)
        Write_X(X, drive, exp, setting)

        country = exp.c
        year4 = exp.y
        day = exp.d
        code = exp.code
        epoch = exp.ep
        logfile = f"{drive}/tomo/{country}/{year4:04d}/{day:03d}/{code}/log.txt"
        with open(logfile, "a") as f:
            print(
                f"SolveProcessEnd : {country} {year4:04d} {day:03d} {code} {epoch:04d} : {datetime.datetime.now()}",
                file=f)

    else:
        N_b = int(math.sqrt(7000 / setting.n_h))
        N_l = int(math.sqrt(7000 / setting.n_h))
        simple_setting = setting.simprize(N_b, N_l)

        s_input = ReadTomoiF(drive, exp)
        s_A = csr_matrix(Gen_A(drive, s_input, simple_setting))
        s_B = Read_B(drive, exp)
        s_H = Gen_H(simple_setting)
        s_Y = Gen_Y(drive, exp, simple_setting)

        sco = float(
            csr_matrix.trace(s_A.T * s_A) / csr_matrix.trace(s_H.T * s_H))
        co = math.sqrt(lamb * sco)
        X = spsolve(s_A.T * s_A + co * co * s_H.T * s_H,
                    s_A.T * s_B + co * co * s_H.T * s_Y)
        X_0 = setting.extension(simple_setting, X)

        A = Read_A(drive, exp)
        B = Read_B(drive, exp)
        H = setting.H
        Y = Gen_Y(drive, exp, setting)

        fco = float(csr_matrix.trace(A.T * A) / setting.trH)
        co = math.sqrt(lamb * fco)
        M0 = A.T * A + co * co * H.T * H
        l_M = M0.shape[0]
        M1 = lil_matrix((l_M, l_M))
        for im in range(l_M):
            M1[im, im] = 1.0 / M0[im, im]
        M1 = M1.tocsr()
        y0 = A.T * B + co * co * H.T * Y
        X, info = bicgstab(A.T * A + co * co * H.T * H,
                           A.T * B + co * co * H.T * Y,
                           x0=X_0,
                           rtol=1.0e-10,
                           maxiter=8000,
                           M=M1,
                           callback=__callback__)
        Write_X(X, drive, exp, setting)
        country = exp.c
        year4 = exp.y
        day = exp.d
        code = exp.code
        epoch = exp.ep
        logfile = f"{drive}/tomo/{country}/{year4:04d}/{day:03d}/{code}/log.txt"
        with open(logfile, "a") as f:
            print(
                f"SolveProcessEnd : {country} {year4:04d} {day:03d} {code} {epoch:04d} : {datetime.datetime.now()}",
                file=f)
        print(exp, num_iter)


def __SolveAfterSecondProblem__(drive,
                                exps: list[EXPERIMENT],
                                setting: SETTING,
                                start,
                                end,
                                tstart=None):
    lamb = setting.lamb
    for idx in range(start, end):
        if idx >= len(exps):
            return
        if idx == 0:
            continue
        exp = exps[idx]
        global num_iter
        num_iter = 0

        A = Read_A(drive, exp)
        B = Read_B(drive, exp)
        H = setting.H
        Y = Gen_Y(drive, exp, setting)

        co = float(csr_matrix.trace(A.T * A) / setting.trH)

        bef_exps = [exps[idx - 1]]
        # print(max(start, idx - 3), min(len(exps), idx + 1))
        bef_exps = exps[max(start, idx - 3):min(len(exps), idx + 1)]
        # print(len(bef_exps))
        bef_exps.reverse()
        # bef_exps = [exps[idx], exps[idx - 1], exps[idx - 2]]
        X_0 = Gen_X0(drive, bef_exps, setting)

        sco = math.sqrt(lamb * co)

        M0 = A.T * A + sco * sco * H.T * H
        l_M = M0.shape[0]
        M1 = lil_matrix((l_M, l_M))
        for im in range(l_M):
            M1[im, im] = 1.0 / M0[im, im]
        M1 = M1.tocsr()
        y0 = A.T * B + sco * sco * H.T * Y
        X, info = bicgstab(M0,
                           y0,
                           x0=X_0,
                           rtol=1.0e-10,
                           maxiter=5000,
                           M=M1,
                           callback=__callback__)
        Write_X(X, drive, exp, setting)
        country = exp.c
        year4 = exp.y
        day = exp.d
        code = exp.code
        epoch = exp.ep
        logfile = f"{drive}/tomo/{country}/{year4:04d}/{day:03d}/{code}/log.txt"
        with open(logfile, "a") as f:
            print(
                f"SolveProcessEnd : {country} {year4:04d} {day:03d} {code} {epoch:04d} : {datetime.datetime.now()}",
                file=f)
        if tstart is not None:
            print(exp, time.time() - tstart, num_iter)
