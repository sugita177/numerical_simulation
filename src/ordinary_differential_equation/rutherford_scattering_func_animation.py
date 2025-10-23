# logistic equation
# runge_kutta_method

import  numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation


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

# 以下はGeminiを利用して作成した
# 間引きレートを定義
skip_steps = 10 # 10ステップごとに描画
N_frames = N // skip_steps # 新しいフレーム総数

# アニメーションのための設定
fig, ax = plt.subplots(figsize=(8.0, 6.0)) # figとaxを分けて取得
ax.set_xlabel("x")
ax.set_xlim(-25, 25)
ax.set_ylabel("y") # 'z'ではなく'y'を使用
ax.set_ylim(-30, 30)
ax.grid(True)
ax.set_title("Charged Particle Motion under Coulomb Force")

# 固定された電荷の位置 (0, 0) を描画
ax.scatter(np.array([0]), np.array([0]), color="r", marker="o", label="Fixed Charge")

# 描画要素の初期化
# 粒子の軌跡全体を格納するLine2Dオブジェクト
line, = ax.plot([], [], color="b", label="Trajectory")
# 現在の粒子の位置をマークするScatterオブジェクト
point, = ax.plot([], [], 'o', color="g", markersize=8, label="Particle Position")

# アニメーションの初期化関数
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# アニメーションの更新関数
# i はフレーム番号
def update(i):
    # 描画するデータのインデックスを計算
    data_index = i * skip_steps

    # 軌跡は data_index までを描画
    x_data = x_list[:data_index+1]
    y_data = y_list[:data_index+1]

    line.set_data(x_data, y_data)
    # 現在の位置も data_index の点を描画
    point.set_data([x_list[data_index]], [y_list[data_index]])

    # タイムスタンプをグラフに追加する場合は以下の行を使用
    # ax.set_title(f"Time: {t_list[data_index]:.2f} s")

    return line, point

# アニメーションの作成
# frames: 全フレーム数
# interval: フレーム間のミリ秒
# blit: True にすると、変更された要素のみを再描画するため高速化される
ani = FuncAnimation(
    fig, 
    update, 
    frames=N_frames, # 間引き後のフレーム数を使用
    init_func=init, 
    interval=10, # 指定した数値のms ごとに更新 (アニメーションの速度を調整)
    blit=True
)

# 凡例を再描画（blit=Trueだとinit_funcで描画された要素以外は消えるため）
ax.legend()

# アニメーションの表示
plt.show()

# アニメーションをファイルに保存する場合は、以下を使用 (例: mp4)
# ani.save('coulomb_motion.mp4', writer='ffmpeg', fps=30)