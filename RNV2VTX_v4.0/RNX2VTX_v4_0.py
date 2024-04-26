import sys

from StationList import __StationList__
from rnx2mdf import __RNX2MDF__
from mdf2bias import __MDF2BIAS__
from mdf2ubs import __MDF2UBS__


def RNX2VTX(drive: str, country: str, year4: int, day: int, setfile: str):
    stations = __StationList__(drive, country, year4, day)
    __RNX2MDF__(drive, country, year4, day, stations, setfile, ver=302)
    __MDF2BIAS__(drive,
                 country,
                 year4,
                 day,
                 setting=setfile,
                 ver="RNX2VTX_ver4.0")
    __MDF2UBS__(drive, country, year4, day, ver="RNX2VTX_ver4.0")


if __name__ == "__main__":
    # args = sys.argv
    drive = "E:"  # args[1]
    country = "jp"  # args[2]
    year4 = 2023  # int(args[3])
    day = 196  # int(args[4])
    setfile = "setting.txt"  # args[5]
    RNX2VTX(drive, country, year4, day, setfile)
