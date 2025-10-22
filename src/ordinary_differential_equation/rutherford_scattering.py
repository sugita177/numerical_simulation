# logistic equation
# runge_kutta_method

import  numpy as np
import matplotlib.pyplot as plt
import math

def coulomb_force_abs(q1, x1, y1, q2=1.0, x2=0.0, y2=0.0, k=1.0) ->float:
    return k * q1 * q2 / ((x1 - x2)**2 + (y1 - y2)**2)

def coulomb_force_x(q1, x1, y1, q2=1.0, x2=0.0, y2=0.0, k=1.0) -> float :
    return coulomb_force_abs(q1, x1, y1, q2, x2, y2, k) * (x1 - x2) / math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def coulomb_force_y(q1, x1, y1, q2=1.0, x2=0.0, y2=0.0, k=1.0) -> float :
    return coulomb_force_abs(q1, x1, y1, q2, x2, y2, k) * (y1 - y2) / math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

t_0: float = 0.0
t_f: float = 100.0
del_t: float = 1e-2
N: int = round((t_f - t_0) / del_t)
t_list = np.zeros(N)
x_list = np.zeros(N)
y_list = np.zeros(N)
vx_list = np.zeros(N)
vy_list = np.zeros(N)

# initial conditions
t_list[0] = t_0
x_list[0] = -20.0
y_list[0] = 2.0
vx_list[0] = 0.8
vy_list[0] = 0.0

# electric charge
q: float = 1.0

for i in range(N-1):
    t: float = t_list[i] + del_t
    x: float = x_list[i]
    y: float = y_list[i]
    vx: float = vx_list[i]
    vy: float = vy_list[i]

    k1 = del_t * vx
    l1 = del_t * vy
    m1 = del_t * coulomb_force_x(q, x, y)
    n1 = del_t * coulomb_force_y(q, x, y)

    k2 = del_t * (vx + m1 / 2.0)
    l2 = del_t * (vy + l1 / 2.0)
    m2 = del_t * coulomb_force_x(q, x + k1 / 2.0, y + l1 / 2.0)
    n2 = del_t * coulomb_force_y(q, x + k1 / 2.0, y + l1 / 2.0)

    k3 = del_t * (vx + m2 / 2.0)
    l3 = del_t * (vy + l2 / 2.0)
    m3 = del_t * coulomb_force_x(q, x + k2 / 2.0, y + l2 / 2.0)
    n3 = del_t * coulomb_force_y(q, x + k2 / 2.0, y + l2 / 2.0)

    k4 = del_t * (vx + m3)
    l4 = del_t * (vy + n3)
    m4 = del_t * coulomb_force_x(q, x + k3, y + l3)
    n4 = del_t * coulomb_force_y(q, x + k3, y + l3)

    x += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    y += (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
    vx += (m1 + 2.0 * m2 + 2.0 * m3 + m4) / 6.0
    vy += (n1 + 2.0 * n2 + 2.0 * n3 + n4) / 6.0

    t_list[i+1] = t
    x_list[i+1] = x
    y_list[i+1] = y
    vx_list[i+1] = vx
    vy_list[i+1] = vy

fig = plt.figure(figsize=(8.0, 6.0))
plt.xlabel("x")
plt.xlim(-25, 25)
plt.ylabel("z")
plt.ylim(-30, 30)

plt.grid(True)
plt.plot(x_list, y_list, color = "b", label = "Runge Kutta")
plt.scatter(np.array([0]), np.array([0]), color = "r", marker="o")
plt.legend()
plt.show()