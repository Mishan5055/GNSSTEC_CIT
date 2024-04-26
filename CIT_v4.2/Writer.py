import os
from f_class import EXPERIMENT, SETTING
import datetime
from scipy.sparse import lil_matrix


def Write_A(A: lil_matrix, drive, exp: EXPERIMENT):
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep

    os.makedirs(f"{drive}/tomoc/{country}/{year4:04d}/{day:03d}/{code}/",
                exist_ok=True)

    _tomoc = f"{drive}/tomoc/{country}/{year4:04d}/{day:03d}/{code}/{ep:04d}.tomoc"

    val = ''
    for i, row in enumerate(A.rows):
        for pos, j in enumerate(row):
            val += f"{i} {j} {A.data[i][pos]}\n"

    with open(_tomoc, "w") as f:
        print(A.shape[0], A.shape[1], file=f)
        print(val, file=f)


def Write_B(B, drive, exp: EXPERIMENT):

    country = exp.c
    year4 = exp.y
    day = exp.d
    ep = exp.ep

    # print(3, exp)

    os.makedirs("{dr}/tomob/{c}/{y:04d}/{d:03d}/".format(dr=drive,
                                                         c=country,
                                                         y=year4,
                                                         d=day),
                exist_ok=True)
    tomob = "{dr}/tomob/{c}/{y:04d}/{d:03d}/{ep:04d}.tomob".format(dr=drive,
                                                                   c=country,
                                                                   y=year4,
                                                                   d=day,
                                                                   ep=ep)

    # print(4, exp)

    with open(tomob, "w") as bf:
        print(len(B), file=bf)
        for i in range(len(B)):
            print(i, B[i], file=bf)


def Write_X(X, drive, exp: EXPERIMENT, setting: SETTING):
    n_all, n_h, n_b, n_l = setting.nbrock()
    a1h, a1b, a1l = setting.plains()

    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep

    os.makedirs("{dr}/tomo/{c}/{y:04}/{d:03}/{code}".format(dr=drive,
                                                            c=country,
                                                            y=year4,
                                                            d=day,
                                                            code=code),
                exist_ok=True)
    op_file = "{dr}/tomo/{c}/{y:04}/{d:03}/{code}/{epc:04d}.tomo".format(
        dr=drive, c=country, y=year4, d=day, code=code, epc=ep)
    ut = ep / 120.0
    dut = datetime.timedelta(hours=ut)
    utsec = dut.seconds
    uth = utsec // 3600
    utm = (utsec - uth * 3600) // 60
    uts = utsec % 60
    with open(op_file, "w") as f:
        print("# Tomography Result", file=f)
        print("# ", file=f)
        print("# RUN BY {prog}".format(prog="tomography.py"), file=f)
        print("# ", file=f)
        print("# UTC : {dt}".format(dt=datetime.datetime.now()), file=f)
        print("# ", file=f)
        print("# Number of Boxel (All area)", file=f)
        print("# Latitude : {nb:03d}".format(nb=n_b), file=f)
        print("# Longitude : {nl:03d}".format(nl=n_l), file=f)
        print("# Height : {nh:03d}".format(nh=n_h), file=f)
        print("# ", file=f)
        print("# *** Plain List ***", file=f)
        print("# Latitude", file=f)
        for ib in range(n_b + 1):
            print("# {b:+06.2f}".format(b=a1b[ib]), file=f)
        print("# Longitude", file=f)
        for il in range(n_l + 1):
            print("# {l:+06.2f}".format(l=a1l[il]), file=f)
        print("# Height", file=f)
        for ih in range(n_h + 1):
            print("# {h:+07.2f}".format(h=a1h[ih]), file=f)
        print("# ", file=f)
        print("# END OF HEADER", file=f)
        print("", file=f)
        for ih in range(n_h):
            for jb in range(n_b):
                for kl in range(n_l):
                    if True:
                        f.write("{teq:+15.13f} ".format(teq=X[kl + jb * n_l +
                                                              ih * n_l * n_b]))
                print("", file=f)
            print("", file=f)
