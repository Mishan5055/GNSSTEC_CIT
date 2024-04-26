import numpy as np
import glob
import f_class as fcl
import os


def __ImportUnbias__(drive, country, year4, day, epochs: list, threshold=0.1):
    top_folder = "{dr}/unbias/{c}/{y4:04d}/{d:03d}".format(dr=drive,
                                                           c=country,
                                                           y4=year4,
                                                           d=day)
    folders = sorted(glob.glob(top_folder + "/*"))
    _inputs: list[fcl.INPUT] = []
    for iep in range(len(epochs)):
        _input = fcl.INPUT()
        _inputs.append(_input)
    for folder in folders:
        print(folder)
        files = sorted(glob.glob(folder + "/*.tec4"))
        for file in files:
            rec_id = file[-9:-5]
            sat_id = file[-13:-10]
            with open(file, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if "Receiver Position" in line:
                        recx = float(line.split()[4]) / 1.0e3
                        recy = float(line.split()[5]) / 1.0e3
                        recz = float(line.split()[6]) / 1.0e3
                        rec = fcl.XYZ(recx, recy, recz)
                    if "# END OF HEADER" in line:
                        break

                while True:
                    line = f.readline()
                    if not line:
                        break

                    ep = round(float(line.split()[0]) * 120.0)
                    if ep in epochs:
                        idx = epochs.index(ep)
                        tec = float(line.split()[1])
                        satx = float(line.split()[2]) / 1e3
                        saty = float(line.split()[3]) / 1e3
                        satz = float(line.split()[4]) / 1e3
                        sat = fcl.XYZ(satx, saty, satz)
                        _inputs[idx].__addF__(tec, sat, rec, sat_id, rec_id)

    os.makedirs(
        "{dr}/tomoiF/{c}/{y4:04d}/{d:03d}/".format(dr=drive,
                                                   c=country,
                                                   y4=year4,
                                                   d=day),
        exist_ok=True,
    )

    os.makedirs(
        "{dr}/tomoiE/{c}/{y4:04d}/{d:03d}/".format(dr=drive,
                                                   c=country,
                                                   y4=year4,
                                                   d=day),
        exist_ok=True,
    )

    for i, ep in enumerate(epochs):
        ep_input: fcl.INPUT = _inputs[i]
        tomoif = "{dr}/tomoiF/{c}/{y4:04d}/{d:03d}/{ep:04d}.tomoif".format(
            dr=drive, c=country, y4=year4, d=day, ep=ep)
        tomoie = "{dr}/tomoiE/{c}/{y4:04d}/{d:03d}/{ep:04d}.tomoie".format(
            dr=drive, c=country, y4=year4, d=day, ep=ep)
        with open(tomoif, "w") as ff:
            nF = ep_input.n_F
            for j in range(nF):
                print(
                    "{tec:+09.4f} {sx:+013.6f} {sy:+013.6f} {sz:+013.6f} {rx:+013.6f} {ry:+013.6f} {rz:+013.6f} {sid} {rid}"
                    .format(
                        tec=ep_input.tecF[j],
                        sx=ep_input.satF[j].x,
                        sy=ep_input.satF[j].y,
                        sz=ep_input.satF[j].z,
                        rx=ep_input.recF[j].x,
                        ry=ep_input.recF[j].y,
                        rz=ep_input.recF[j].z,
                        sid=ep_input.sat_idF[j],
                        rid=ep_input.rec_idF[j],
                    ),
                    file=ff,
                )

        with open(tomoie, "w") as fe:
            nE = ep_input.n_E
            for j in range(nE):
                print(
                    "{tec:+09.4f} {sx:+013.6f} {sy:+013.6f} {sz:+013.6f} {rx:+013.6f} {ry:+013.6f} {rz:+013.6f} {sid} {rid}"
                    .format(
                        tec=ep_input.tecE[j],
                        sx=ep_input.satE[j].x,
                        sy=ep_input.satE[j].y,
                        sz=ep_input.satE[j].z,
                        rx=ep_input.recE[j].x,
                        ry=ep_input.recE[j].y,
                        rz=ep_input.recE[j].z,
                        sid=ep_input.sat_idE[j],
                        rid=ep_input.rec_idE[j],
                    ),
                    file=fe,
                )

    for i, ep in enumerate(epochs):
        print("{ep:04d} {nE:05d}+{nF:05d}/{n:05d}".format(
            ep=ep,
            nE=_inputs[i].n_E,
            nF=_inputs[i].n_F,
            n=_inputs[i].n_E + _inputs[i].n_F,
        ))


if __name__ == "__main__":
    drive = "E:"
    country = "jp"
    year4 = 2023
    days = [196]
    for day in days:
        epochs = np.arange(1200, 2200, 2).tolist()
        __ImportUnbias__(drive, country, year4, day, epochs)
