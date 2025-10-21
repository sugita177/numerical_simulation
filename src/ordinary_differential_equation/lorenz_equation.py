# lorenz_equation
# runge_kutta_method

import numpy as np
import matplotlib.pyplot as plt

def lorenz_x(x: float, y: float, sigma: float = 10) -> float :
    return sigma * (y - x)

def lorenz_y(x: float, y: float, z: float, rho: float = 28) -> float :
    return x * (rho - z) - y

def lorenz_z(x: float, y: float, z: float, beta: float = 8.0/3.0) -> float :
    return x * y - beta * z

t_0: float = 0.0
t_f: float = 100.0
del_t: float = 1e-2
N: int = round((t_f - t_0) / del_t)
t_list = np.zeros(N)
x_list = np.zeros(N)
y_list = np.zeros(N)
z_list = np.zeros(N)
t_list[0] = t_0
x_list[0] = 10.0
y_list[0] = 10.0
z_list[0] = 10.0

for i in range(N-1):
    t: float = t_list[i] + del_t
    x: float = x_list[i]
    y: float = y_list[i]
    z: float = z_list[i]

    k1 = del_t * lorenz_x(x, y)
    l1 = del_t * lorenz_y(x, y, z)
    m1 = del_t * lorenz_z(x, y, z)

    k2 = del_t * lorenz_x(x + k1 / 2.0, y + l1 / 2.0)
    l2 = del_t * lorenz_y(x + k1 / 2.0, y + l1 / 2.0, z + m1 / 2.0)
    m2 = del_t * lorenz_z(x + k1 / 2.0, y + l1 / 2.0, z + m1 / 2.0)

    k3 = del_t * lorenz_x(x + k2 / 2.0, y + l2 / 2.0)
    l3 = del_t * lorenz_y(x + k2 / 2.0, y + l2 / 2.0, z + m2 / 2.0)
    m3 = del_t * lorenz_z(x + k2 / 2.0, y + l2 / 2.0, z + m2 / 2.0)

    k4 = del_t * lorenz_x(x + k3, y + l3)
    l4 = del_t * lorenz_y(x + k3, y + l3, z + m3)
    m4 = del_t * lorenz_z(x + k3, y + l3, z + m3)

    x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    y += (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    z += (m1 + 2.0 * m2 + 2.0 * m3 + m4) / 6.0
    t_list[i+1] = t
    x_list[i+1] = x
    y_list[i+1] = y
    z_list[i+1] = z

fig = plt.figure(figsize=(8.0, 6.0))
plt.xlabel("x")
plt.xlim(-20, 20)
plt.ylabel("z")
plt.ylim(0, 50)

plt.grid(True)
plt.plot(x_list, z_list, color = "b", label = "Runge Kutta")
plt.legend()
plt.show()