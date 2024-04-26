import os
import glob
import datetime

from f_class import XYZ

BIAS_CODE = {"GPS": -1, "GLONASS": -1, "GALILEO": -1, "QZSS": -1}

SATELLITE_BIAS = {}
RECEIVER_BIAS = {}


def __MDF2UBS__(drive: str, country: str, year4: int, day: int, ver: str):
    global BIAS_CODE, SATELLITE_BIAS, RECEIVER_BIAS
    vtecmap_file = f"{drive}/vtecmap/{country}/{year4:04d}/{day:03d}/{ver}.vtecmap"
    with open(vtecmap_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                break
            if "# BIAS CODE" in line:
                line = f.readline()
                BIAS_CODE["GPS"] = int(line.split()[2])
                line = f.readline()
                BIAS_CODE["GLONASS"] = int(line.split()[2])
                line = f.readline()
                BIAS_CODE["GALILEO"] = int(line.split()[2])
                line = f.readline()
                BIAS_CODE["QZSS"] = int(line.split()[2])

        while True:
            line = f.readline()
            if not line:
                break
            if "## SATELLITE BIAS" in line:
                while True:
                    line = f.readline()
                    sline = line.split()
                    if len(sline) < 2:
                        break
                    sat = sline[0]
                    bias = float(sline[1])
                    if sat in SATELLITE_BIAS:
                        pass
                    else:
                        SATELLITE_BIAS[sat] = bias

            if "## RECEIVER BIAS" in line:
                while True:
                    line = f.readline()
                    sline = line.split()
                    if len(sline) < 1:
                        break
                    rec = sline[0]
                    biases = sline[1:]
                    if rec in RECEIVER_BIAS:
                        pass
                    else:
                        # print(rec, biases)
                        RECEIVER_BIAS[rec] = biases

    sat_folders = sorted(
        glob.glob(f"{drive}/mdf/{country}/{year4:04d}/{day:03d}/*"))

    for sat_folder in sat_folders:
        sat_id = sat_folder[-3:]
        files = sorted(glob.glob(sat_folder + "/*.mdf4"))
        for file in files:
            rec_id = file[-9:-5]
            sat_bias = SATELLITE_BIAS[sat_id]
            if "G" in sat_id:
                rec_bias = float(RECEIVER_BIAS[rec_id][BIAS_CODE["GPS"]])
            if "R" in sat_id:
                rec_bias = float(RECEIVER_BIAS[rec_id][BIAS_CODE["GLONASS"]])
            if "E" in sat_id:
                rec_bias = float(RECEIVER_BIAS[rec_id][BIAS_CODE["GALILEO"]])
            if "J" in sat_id:
                rec_bias = float(RECEIVER_BIAS[rec_id][BIAS_CODE["QZSS"]])

            with open(file, "r") as f:
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
                        rec_pos = XYZ(X, Y, Z)

                times = []
                stecs = []
                sats = []
                zeniths = []

                while True:
                    line = f.readline()
                    if not line:
                        break
                    time = float(line.split()[0])
                    stec = float(line.split()[1])
                    sx = float(line.split()[4])
                    sy = float(line.split()[5])
                    sz = float(line.split()[6])
                    sat = XYZ(sx, sy, sz)
                    zenith = float(line.split()[7])
                    times.append(time)
                    stecs.append(stec)
                    sats.append(sat)
                    zeniths.append(zenith)

            L = len(times)

            os.makedirs(
                f"{drive}/unbias/{country}/{year4:04d}/{day:03d}/{sat_id}/",
                exist_ok=True)
            ubs_file = f"{drive}/unbias/{country}/{year4:04d}/{day:03d}/{sat_id}/{rec_id}.tec4"

            with open(ubs_file, "w") as f:
                print(f"# PRN {sat_id}", file=f)
                print(f"#", file=f)
                print(f"# RNX2VTX : RNX2VTX_ver4.0", file=f)
                print(f"# UTCTIME : {datetime.datetime.now()}", file=f)
                print(f"#", file=f)
                rx = rec_pos.x
                ry = rec_pos.y
                rz = rec_pos.z
                print(
                    f"# Receiver Position : {rx:014.4f} {ry:014.4f} {rz:014.4f}",
                    file=f)
                print(f"#", file=f)
                print(f"# Receiver bias : {rec_bias:07.4f}", file=f)
                print(f"# Satellite bias : {sat_bias:07.4f}", file=f)
                print(f"#", file=f)
                print(f"# END OF HEADER", file=f)

                for iepoch in range(L):
                    print(
                        f"{times[iepoch]:07.4f} {stecs[iepoch]-rec_bias-sat_bias:07.4f} {sats[iepoch].x:014.4f} {sats[iepoch].y:014.4f} {sats[iepoch].z:014.4f} {zeniths[iepoch]:07.4f}",
                        file=f)

    return
