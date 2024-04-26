import math
import numpy as np
from scipy.sparse import csr_matrix

rf = 1.0 / 298.257223563
ra = 6378.1370
rb = ra * (1.0 - rf)
re = math.sqrt((ra * ra - rb * rb) / (ra * ra))


class BLH:
    # b,l...[degree]
    # h...[km]
    b: float = 0.0
    l: float = 0.0
    h: float = 0.0

    def __init__(self, b, l, h):
        self.b = b
        self.l = l
        self.h = h

    def to_XYZ(self):
        answer = XYZ(0.0, 0.0, 0.0)
        n = ra / math.sqrt(1.0 - re * re * math.sin(math.radians(self.b)) *
                           math.sin(math.radians(self.b)))
        answer.x = (n+self.h)*math.cos(math.radians(self.b)) * \
            math.cos(math.radians(self.l))
        answer.y = (n+self.h)*math.cos(math.radians(self.b)) * \
            math.sin(math.radians(self.l))
        answer.z = (
            (1 - re * re) * n + self.h) * math.sin(math.radians(self.b))
        return answer

    def __str__(self):
        return "[ B: " + str(self.b) + " L: " + str(self.l) + " H: " + str(
            self.h) + " ]"


class XYZ:

    # x,y,z...[km]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_BLH(self):
        X = float(self.x)
        Y = float(self.y)
        Z = float(self.z)
        answer = BLH(0.0, 0.0, 0.0)
        # 1
        p = math.sqrt(X * X + Y * Y)
        h = ra * ra - rb * rb
        t = math.atan2(Z * ra, p * rb)  # rad
        answer.l = math.degrees(math.atan2(Y, X))  # deg
        # 2
        answer.b = math.degrees(
            math.atan2((ra * rb * Z + ra * h * math.sin(t)**3),
                       (ra * rb * p - rb * h * math.cos(t)**3)))  # deg
        # 3
        n = ra / math.sqrt(1 - re * re * math.sin(math.radians(answer.b)) *
                           math.sin(math.radians(answer.b)))
        # 4
        answer.h = p / math.cos(math.radians(answer.b)) - n
        return answer

    def __str__(self):
        return "[ X: " + str(self.x) + " Y: " + str(self.y) + " Z: " + str(
            self.z) + " ]"

    def __add__(self, other):
        spo = XYZ(self.x + other.x, self.y + other.y, self.z + other.z)
        return spo

    def __sub__(self, other):
        smo = XYZ(self.x - other.x, self.y - other.y, self.z - other.z)
        return smo

    def __mul__(self, other):
        sto = XYZ(self.x * other, self.y * other, self.z * other)
        return sto

    def __rmul__(self, other):
        sto = XYZ(self.x * other, self.y * other, self.z * other)
        return sto

    def L2(self) -> float:
        siz = self.x**2 + self.y**2 + self.z**2
        return math.sqrt(siz)


class EXPERIMENT:
    c: str = ""
    y: int = 0
    d: int = 0
    code: str = ""
    ep: int = 0

    def __init__(self, c, y, d, code, ep):
        self.c = c
        self.y = y
        self.d = d
        self.code = code
        self.ep = ep

    def __str__(self):
        return self.c + " " + str(self.y) + " / " + str(
            self.d) + " " + self.code + " : " + str(self.ep)


class SETTING:
    a1b: np.ndarray = np.array([])
    a1l: np.ndarray = np.array([])
    a1h: np.ndarray = np.array([])
    a2b: np.ndarray = np.array([])
    a2l: np.ndarray = np.array([])
    a2h: np.ndarray = np.array([])

    n_h: int = 0
    n_b: int = 0
    n_l: int = 0
    n_all: int = 0
    lamb: float = 3.0e8
    cof: np.ndarray = np.array([])
    alpha: float = 1.2

    m_H: float = 0.0
    M_H: float = 0.0
    m_B: float = 0.0
    M_B: float = 0.0
    m_L: float = 0.0
    M_L: float = 0.0

    H: csr_matrix = csr_matrix((1, 1))
    trH: float = 0.0

    def __init__(self, a1h: np.ndarray, a1b: np.ndarray, a1l: np.ndarray,
                 lamb: float, cof: np.ndarray, alpha: float):
        self.a1h = a1h
        self.a1b = a1b
        self.a1l = a1l
        self.lamb = lamb
        self.cof = cof
        self.alpha = alpha

        self.n_h = a1h.shape[0] - 1
        self.n_b = a1b.shape[0] - 1
        self.n_l = a1l.shape[0] - 1
        self.n_all = self.n_h * self.n_b * self.n_l

        self.a2h = np.full((self.n_h), 0.0, dtype=float)
        self.a2b = np.full((self.n_b), 0.0, dtype=float)
        self.a2l = np.full((self.n_l), 0.0, dtype=float)
        for ih in range(self.n_h):
            self.a2h[ih] = 0.5 * (self.a1h[ih] + self.a1h[ih + 1])
        for ib in range(self.n_b):
            self.a2b[ib] = 0.5 * (self.a1b[ib] + self.a1b[ib + 1])
        for il in range(self.n_l):
            self.a2l[il] = 0.5 * (self.a1l[il] + self.a1l[il + 1])

        self.m_H = np.min(self.a1h)
        self.M_H = np.max(self.a1h)
        self.m_B = np.min(self.a1b)
        self.M_B = np.max(self.a1b)
        self.m_L = np.min(self.a1l)
        self.M_L = np.max(self.a1l)

    def __initH__(self, H: csr_matrix):
        self.H = H
        self.trH = csr_matrix.trace(self.H.T * self.H)

    def nbrock(self) -> tuple[int, int, int, int]:
        """_summary_
        settingのブロック数を返します。 \n
        return : n_all, n_h, n_b, n_l \n
        Returns: \n
            tuple[int, int, int, int]: _description_
        """
        return [self.n_all, self.n_h, self.n_b, self.n_l]

    def plains(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_
        settingに設定された境界を返します。\n
        return : a1h, a1b, a1l \n
        Returns: \n
            tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        return [self.a1h, self.a1b, self.a1l]

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        return : a2h, a2b, a2l
        """
        return [self.a2h, self.a2b, self.a2l]

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return [self.m_H, self.M_H, self.m_B, self.M_B, self.m_L, self.M_L]

    def simprize(self, N_b, N_l):
        l2b = np.full((self.n_b), 0.0, dtype=float)
        l2l = np.full((self.n_l), 0.0, dtype=float)
        for ib in range(self.n_b):
            l2b[ib] = self.a1b[ib + 1] - self.a1b[ib]
        for il in range(self.n_l):
            l2l[il] = self.a1l[il + 1] - self.a1l[il]
        b2b = np.full((self.n_b + 1), True, dtype=bool)
        b2l = np.full((self.n_l + 1), True, dtype=bool)
        loop = 0
        while np.sum(b2b) > N_b:
            argMIN = -1
            MIN = 100.0
            lower = 0
            upper = 1
            while upper < self.n_b + 1:
                if b2b[upper]:
                    dif = self.a1b[upper] - self.a1b[lower]
                    if dif < MIN:
                        MIN = dif
                        argMIN = upper
                    lower = upper
                upper += 1
            b2b[argMIN] = False
            # print(loop, end=": ")
            # for ib in range(self.n_b+1):
            #     if b2b[ib]:
            #         print(self.a1b[ib], end=" ")
            # print("")
            loop += 1
        loop = 0
        while np.sum(b2l) > N_l:
            argMIN = -1
            MIN = 100.0
            lower = 0
            upper = 1
            while upper < self.n_l + 1:
                if b2l[upper]:
                    dif = self.a1l[upper] - self.a1l[lower]
                    if dif < MIN:
                        MIN = dif
                        argMIN = upper
                    lower = upper
                upper += 1
            b2l[argMIN] = False
            # print(loop, end=": ")
            # for il in range(self.n_l+1):
            #     if b2l[il]:
            #         print(self.a1l[il], end=" ")
            # print("")
            loop += 1
        simple_a1b = []
        simple_a1l = []
        for ib in range(self.n_b + 1):
            if b2b[ib]:
                simple_a1b.append(self.a1b[ib])
        for il in range(self.n_l + 1):
            if b2l[il]:
                simple_a1l.append(self.a1l[il])

        simprize_prob = SETTING(self.a1h, np.array(simple_a1b),
                                np.array(simple_a1l), self.lamb, self.cof,
                                self.alpha)
        return simprize_prob

    def extension(self, simple, X) -> np.ndarray:
        X_0 = np.full((self.n_all), 0.0, dtype=float)
        sa1h, sa1b, sa1l = simple.plains()
        for ih in range(self.n_h):
            sh = np.searchsorted(sa1h, self.a2h[ih]) - 1
            for jb in range(self.n_b):
                sb = np.searchsorted(sa1b, self.a2b[jb]) - 1
                for kl in range(self.n_l):
                    sl = np.searchsorted(sa1l, self.a2l[kl]) - 1
                    idx = kl + jb * self.n_l + ih * self.n_b * self.n_l
                    sidx = sl + sb * simple.n_l + sh * simple.n_b * simple.n_l
                    X_0[idx] = X[sidx]

        return X_0


class INPUT:
    n_E: int = 0
    tecE: list[float] = []
    satE: list[XYZ] = []
    recE: list[XYZ] = []
    sat_idE: list[str] = []
    rec_idE: list[str] = []

    n_F: int = 0
    tecF: list[float] = []
    satF: list[XYZ] = []
    recF: list[XYZ] = []
    sat_idF: list[str] = []
    rec_idF: list[str] = []

    def __init__(self):
        self.n_E = 0
        self.tecE = []
        self.satE = []
        self.recE = []
        self.sat_idE = []
        self.rec_idE = []

        self.n_F = 0
        self.tecF = []
        self.satF = []
        self.recF = []
        self.sat_idF = []
        self.rec_idF = []
        return

    def __addE__(self, tec: float, sat: XYZ, rec: XYZ, id_sat: str,
                 id_rec: str):
        self.tecE.append(tec)
        self.satE.append(sat)
        self.recE.append(rec)
        self.sat_idE.append(id_sat)
        self.rec_idE.append(id_rec)
        self.n_E += 1

    def __addF__(self, tec: float, sat: XYZ, rec: XYZ, id_sat: str,
                 id_rec: str):
        self.tecF.append(tec)
        self.satF.append(sat)
        self.recF.append(rec)
        self.sat_idF.append(id_sat)
        self.rec_idF.append(id_rec)
        self.n_F += 1

    def __size__(self):
        return self.n_F, self.n_E
