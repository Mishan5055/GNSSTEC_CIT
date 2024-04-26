import numpy as np
import matplotlib.pyplot as plt

file = "E:/iteration/jp/2023/013/CIT40_TID_lam=3e8_alpha=8e-1_beta=adapt/1202.txt"

N = 200
res0 = np.full((N), 0.0, dtype=float)
res1 = np.full((N), 0.0, dtype=float)
res2 = np.full((N), 0.0, dtype=float)

with open(file, "r") as f:
    for i in range(N):
        line = f.readline()
        if not line:
            break
        res0[i] = float(line.split()[1])
        res1[i] = float(line.split()[2])
        res2[i] = float(line.split()[3])

fig, ax = plt.subplots(1, 1, squeeze=False)
ax2 = ax[0, 0].twinx()

ax[0, 0].plot(res0, color="black", label="all")
ax[0, 0].plot(res1, color="red", label="1st")
ax2.plot(res2, color="blue", label="2nd")
ax[0, 0].set_ylim(0, 600000)
ax2.set_ylim(100000, 500000)
plt.legend()
plt.show()
