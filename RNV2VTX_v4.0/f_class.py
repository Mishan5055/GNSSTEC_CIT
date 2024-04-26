import math

rf = 1.0 / 298.257223563
ra = 6378137.0
rb = ra * (1.0 - rf)
re = math.sqrt((ra * ra - rb * rb) / (ra * ra))


class BLH:
    # b,l...[degree]
    # h...[km]
    b: float = 0.0
    l: float = 0.0
    h: float = 0.0

    def __init__(self, b=0.0, l=0.0, h=0.0):
        self.b = b
        self.l = l
        self.h = h

    def to_XYZ(self):
        answer = XYZ()
        n = ra / math.sqrt(1.0 - re * re * math.sin(math.radians(self.b)) *
                           math.sin(math.radians(self.b)))
        answer.x = ((n + self.h) * math.cos(math.radians(self.b)) *
                    math.cos(math.radians(self.l)))
        answer.y = ((n + self.h) * math.cos(math.radians(self.b)) *
                    math.sin(math.radians(self.l)))
        answer.z = (
            (1 - re * re) * n + self.h) * math.sin(math.radians(self.b))
        return answer

    def __str__(self):
        return ("[ B: " + str(self.b) + " L: " + str(self.l) + " H: " +
                str(self.h) + " ]")


class XYZ:
    # x,y,z...[km]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def to_BLH(self):
        # print(type(self))
        # print(self.x)
        X = float(self.x)
        Y = float(self.y)
        Z = float(self.z)
        answer = BLH()
        # 1
        # print(type(X))
        # print(type(Y))
        p = math.sqrt(X * X + Y * Y)
        h = ra * ra - rb * rb
        t = math.atan2(Z * ra, p * rb)  # rad
        answer.l = math.degrees(math.atan2(Y, X))  # deg
        # 2
        answer.b = math.degrees(
            math.atan2(
                (ra * rb * Z + ra * h * math.sin(t)**3),
                (ra * rb * p - rb * h * math.cos(t)**3),
            ))  # deg
        # 3
        n = ra / math.sqrt(1 - re * re * math.sin(math.radians(answer.b)) *
                           math.sin(math.radians(answer.b)))
        # 4
        answer.h = p / math.cos(math.radians(answer.b)) - n
        return answer

    def __str__(self):
        return ("[ X: " + str(self.x) + " Y: " + str(self.y) + " Z: " +
                str(self.z) + " ]")

    def __add__(self, other):
        answer = XYZ(self.x + other.x, self.y + other.y, self.z + other.z)
        return answer

    def __sub__(self, other):
        answer = XYZ(self.x - other.x, self.y - other.y, self.z - other.z)
        return answer

    def L2(self) -> float:
        siz = self.x**2 + self.y**2 + self.z**2
        return math.sqrt(siz)


class STECRecord:
    stec: float = 0.0
    sat: XYZ
    rec: XYZ
    sat_id: str
    rec_id: str

    def __init__(self,
                 stec: float = None,
                 sat: XYZ = None,
                 rec: XYZ = None,
                 sat_id=None,
                 rec_id=None):
        if stec is None:
            self.stec = 0.0
        else:
            self.stec = stec

        if sat is None:
            self.sat = XYZ(0, 0, 0)
        else:
            self.sat = sat

        if rec is None:
            self.rec = XYZ(0, 0, 0)
        else:
            self.rec = rec

        if sat_id is None:
            self.sat_id = ""
        else:
            self.sat_id = sat_id

        if rec_id is None:
            self.rec_id = ""
        else:
            self.rec_id = rec_id
