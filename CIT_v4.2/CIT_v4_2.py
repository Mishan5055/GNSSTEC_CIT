import time
import datetime
import os

import numpy as np

from concurrent.futures import ProcessPoolExecutor

import f_class as fcl
from Reader import Read_Setting
from Generator import Gen_H
from Solver import __SettingProcess__, __SolveFirstProblem__, __SolveAfterSecondProblem__


def CIT(drive, exps: list[fcl.EXPERIMENT], fsetting: str):

    exp = exps[0]
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code

    os.makedirs(f"{drive}/tomo/{country}/{year4:04d}/{day:03d}/{code}/",
                exist_ok=True)
    logfile = f"{drive}/tomo/{country}/{year4:04d}/{day:03d}/{code}/log.txt"
    with open(logfile, "a") as f:
        print(f"ProgramStart : {datetime.datetime.now()}", file=f)

    start = time.time()

    _setting = Read_Setting(fsetting)

    H = Gen_H(_setting)
    # ----------ここまでOK-------------------
    _setting.__initH__(H)

    print("End init Process", time.time() - start)

    # Setting Process
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(__SettingProcess__, drive, exp, _setting)
            for exp in exps
        ]

    print("End Setting Process", time.time() - start)

    # # Solve First Problem
    __SolveFirstProblem__(drive, exps[0], _setting)

    print("Solved First Problem", time.time() - start)

    # # Solve All Other Problem
    __SolveAfterSecondProblem__(drive,
                                exps,
                                _setting,
                                start=0,
                                end=1000,
                                tstart=start)

    print("Process Time :", time.time() - start)


if __name__ == "__main__":
    drive = "E:"
    country = "jp"
    year4 = 2023
    code = "CIT41_TID_lam=3e8_alpha=12e-1_Season"
    days = [17]
    start = [1000]
    end = [2100]
    for i, day in enumerate(days):
        exps = []
        for ep in range(start[i], end[i], 2):
            exp = fcl.EXPERIMENT(country, year4, day, code, ep)
            exps.append(exp)
        CIT(drive, exps, fsetting="setting4.txt")
