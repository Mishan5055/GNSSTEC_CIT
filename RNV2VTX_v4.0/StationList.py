import glob
import os


def __StationList__(drive, country, year4, day) -> list[str]:
    folder = "{dr}/tec/{c}/{y4:04d}/{d:03d}".format(dr=drive,
                                                    c=country,
                                                    y4=year4,
                                                    d=day)
    files = glob.glob(folder + "/*")

    stationlist = set()

    for file in files:
        year2 = int(file[-3:-1])
        doy = int(file[-8:-5])
        station = file[-12:-8]
        if year4 % 100 == year2 and day == doy:
            stationlist.add(station)
    return sorted(list(stationlist))
