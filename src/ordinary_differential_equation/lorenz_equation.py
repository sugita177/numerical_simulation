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

# 以下はGeminiを使用して作成

# 2次元グラフの描画
fig_2d, axes = plt.subplots(1, 3, figsize=(15, 5)) # 1行3列のサブプロットを作成

# -----------------
# 1. xy 平面グラフ
# -----------------
axes[0].plot(x_list, y_list, color = "r")
axes[0].set_title("xy plane")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(True)

# -----------------
# 2. yz 平面グラフ
# -----------------
axes[1].plot(y_list, z_list, color = "g")
axes[1].set_title("yz plane")
axes[1].set_xlabel("y")
axes[1].set_ylabel("z")
axes[1].grid(True)

# -----------------
# 3. xz 平面グラフ
# -----------------
axes[2].plot(x_list, z_list, color = "b")
axes[2].set_title("xz plane")
axes[2].set_xlabel("x")
axes[2].set_ylabel("z")
axes[2].grid(True)

plt.tight_layout() # サブプロット間のスペースを自動調整
# plt.show() # 3次元グラフと一緒に表示するためにコメントアウトするか、後に移動

# 3次元グラフの描画
fig_3d = plt.figure(figsize=(10, 8))
# 3Dプロットを追加するためにadd_subplotにprojection='3d'を指定
ax_3d = fig_3d.add_subplot(111, projection='3d') 

# -----------------
# 3次元プロット
# -----------------
# plotメソッドは、x, y, zの3つのリストを取ります
ax_3d.plot(x_list, y_list, z_list, color='purple')

# 軸ラベルの設定
ax_3d.set_xlabel("X Axis")
ax_3d.set_ylabel("Y Axis")
ax_3d.set_zlabel("Z Axis")
ax_3d.set_title("Lorenz Attractor (3D)") # ローレンツアトラクターの図を描画 

# 視点の設定 (任意)
# ax_3d.view_init(elev=20, azim=120)

plt.tight_layout() # グラフのレイアウトを調整
plt.show() # ここで全てのグラフを表示