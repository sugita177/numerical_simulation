# lorenz_equation
# runge_kutta_method

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation 


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
# ==================================
# グラフ描画・アニメーション部分 (修正)
# ==================================

# 描画するフレーム数を調整 (軌跡の伸びる速さ)
# N = 10000 フレーム総数
# step = 10 なら 1000 フレームになる
step = 10 
N_frames = N // step 

# 3次元グラフの設定
fig_3d = plt.figure(figsize=(10, 8))
# ⚠ 3Dプロットにはprojection='3d'が必要です
ax_3d = fig_3d.add_subplot(111, projection='3d')

# 3次元プロットの初期化 (最初は空の線)
# これがアニメーションで更新されるラインオブジェクトになります。
line, = ax_3d.plot([], [], [], color='lime', linewidth=1.0) 

# 軸ラベルとタイトルの設定
ax_3d.set_xlabel("X Axis")
ax_3d.set_ylabel("Y Axis")
ax_3d.set_zlabel("Z Axis")
ax_3d.set_title(f"Lorenz Attractor Trace (Step={step})")

# 軸の限界を固定 (アニメーション中にグラフの範囲が変わらないようにする)
# これにより、軌跡が画面外に出てしまうのを防ぎ、見やすくなります。
ax_3d.set_xlim(x_list.min(), x_list.max())
ax_3d.set_ylim(y_list.min(), y_list.max())
ax_3d.set_zlim(z_list.min(), z_list.max())


def update_trace(frame):
    # frame * step までのデータポイントを描画
    # frame=0: 0まで, frame=1: stepまで, ...
    end_index = frame * step
    
    # プロットデータを更新
    line.set_data(x_list[:end_index], y_list[:end_index])
    line.set_3d_properties(z_list[:end_index])
    
    # 視点をゆっくり回転させることも可能です (任意)
    # ax_3d.view_init(elev=30, azim=frame * 0.1) 
    
    return line,

# FuncAnimationを使ってアニメーションを作成
ani_trace = animation.FuncAnimation(
    fig_3d, 
    update_trace, 
    frames=N_frames,        # フレーム数
    interval=10,            # フレーム間の遅延 (ミリ秒)
    blit=False,             # 3Dプロットでは通常False
    repeat=False            # 軌跡を描き終わったら停止
)

plt.show()