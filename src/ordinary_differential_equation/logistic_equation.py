# logistic equation
# runge_kutta_method

import numpy as np
import matplotlib.pyplot as plt


def logistic(x: float, r: float = 1.2, K: float = 3.0) -> float:
    return r * x * (1 - x / K)


def exact_ans(t: float, x0: float, r: float = 1.2, K: float = 3.0) -> float:
    return K / (1.0 + (K / x0 - 1.0) * np.exp(-r * t))


N: int = 20000
h: float = 1.0e-3
t: float = 0
t_list = np.array([])
x0: float = 0.1
x: float = x0
x_list = np.array([])
exac_list = np.array([])

t_list = np.append(t_list, t)
x_list = np.append(x_list, x)
exac_list = np.append(exac_list, x)

for i in range(N):
    t = i * h
    k1 = h * logistic(x)
    k2 = h * logistic(x + k1 / 2.0)
    k3 = h * logistic(x + k2 / 2.0)
    k4 = h * logistic(x + k3)
    x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    t_list = np.append(t_list, t+h)
    x_list = np.append(x_list, x)
    exac_list = np.append(exac_list, exact_ans(t+h, x0))

fig = plt.figure(figsize=(8.0, 6.0))
plt.xlabel("time")
plt.xlim(0, 20.0)
plt.ylabel("number of indivisuals")
plt.ylim(-1.0, 4.0)

plt.grid(True)
plt.plot(t_list, exac_list, color="k", label="exact_ans")
plt.scatter(t_list, x_list, color="b", label="Runge Kutta", marker="o")
plt.legend()
plt.show()
