# two body problem
# runge kutta method

import  numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation

# eps softening parameter
def gravitational_force_abs(m1, x1, y1, m2=1.0, x2=0.0, y2=0.0, G=1.0, eps=1e-1) ->float:
    return G * m1 * m2 / ((x1 - x2)**2 + (y1 - y2)**2 + eps**2)

def gravitational_force_x(m1, x1, y1, m2=1.0, x2=0.0, y2=0.0, G=1.0, eps=1e-1) -> float :
    return (-1.0) * gravitational_force_abs(m1, x1, y1, m2, x2, y2, G, eps) * (x1 - x2) / math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + eps**2)

def gravitational_force_y(m1, x1, y1, m2=1.0, x2=0.0, y2=0.0, G=1.0, eps=1e-1) -> float :
    return (-1.0) * gravitational_force_abs(m1, x1, y1, m2, x2, y2, G, eps) * (y1 - y2) / math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + eps**2)

t_0: float = 0.0
t_f: float = 100.0
del_t: float = 1e-2
N: int = round((t_f - t_0) / del_t)
t_list = np.zeros(N)
x1_list = np.zeros(N)
y1_list = np.zeros(N)
vx1_list = np.zeros(N)
vy1_list = np.zeros(N)
x2_list = np.zeros(N)
y2_list = np.zeros(N)
vx2_list = np.zeros(N)
vy2_list = np.zeros(N)

# initial conditions
t_list[0] = t_0
x1_list[0] = -10.0
y1_list[0] = 0.0
vx1_list[0] = 0.0
vy1_list[0] = 4.0
x2_list[0] = 10.0
y2_list[0] = 0.0
vx2_list[0] = 0.0
vy2_list[0] = 0.0

# mass
m1: float = 1.0
m2: float = 1000.0

for i in range(N-1):
    t: float = t_list[i] + del_t
    x1: float = x1_list[i]
    y1: float = y1_list[i]
    vx1: float = vx1_list[i]
    vy1: float = vy1_list[i]
    x2: float = x2_list[i]
    y2: float = y2_list[i]
    vx2: float = vx2_list[i]
    vy2: float = vy2_list[i]


    fx11 = gravitational_force_x(m1, x1, y1, m2, x2, y2)
    fy11 = gravitational_force_y(m1, x1, y1, m2, x2, y2)
    k11 = del_t * vx1
    l11 = del_t * vy1
    k12 = del_t * vx2
    l12 = del_t * vy2
    r11 = del_t * fx11 / m1
    s11 = del_t * fy11 / m1
    r12 = del_t * (-1.0) * fx11 / m2
    s12 = del_t * (-1.0) * fy11 / m2

    fx21 = gravitational_force_x(m1, x1 + k11 / 2.0, y1 + l11 / 2.0, m2, x2 + k12 / 2.0, y2 + l12 / 2.0)
    fy21 = gravitational_force_y(m1, x1 + k11 / 2.0, y1 + l11 / 2.0, m2, x2 + k12 / 2.0, y2 + l12 / 2.0)
    k21 = del_t * (vx1 + r11 / 2.0)
    l21 = del_t * (vy1 + s11 / 2.0)
    k22 = del_t * (vx2 + r12 / 2.0)
    l22 = del_t * (vy2 + s12 / 2.0)
    r21 = del_t * fx21 / m1
    s21 = del_t * fy21 / m1
    r22 = del_t * fx21 * (-1.0) / m2
    s22 = del_t * fy21 * (-1.0) / m2

    fx31 = gravitational_force_x(m1, x1 + k21 / 2.0, y1 + l21 / 2.0, m2, x2 + k22 / 2.0, y2 + l22 / 2.0)
    fy31 = gravitational_force_y(m1, x1 + k21 / 2.0, y1 + l21 / 2.0, m2, x2 + k22 / 2.0, y2 + l22 / 2.0)
    k31 = del_t * (vx1 + r21 / 2.0)
    l31 = del_t * (vy1 + s21 / 2.0)
    k32 = del_t * (vx2 + r22 / 2.0)
    l32 = del_t * (vy2 + s22 / 2.0)
    r31 = del_t * fx31 / m1
    s31 = del_t * fy31 / m1
    r32 = del_t * fx31 * (-1.0) / m2
    s32 = del_t * fy31 * (-1.0) / m2

    fx41 = gravitational_force_x(m1, x1 + k31, y1 + l31, m2, x2 + k32, y2 + l32)
    fy41 = gravitational_force_y(m1, x1 + k31, y1 + l31, m2, x2 + k32, y2 + l32)
    k41 = del_t * (vx1 + r31)
    l41 = del_t * (vy1 + s31)
    k42 = del_t * (vx2 + r32)
    l42 = del_t * (vy2 + s32)
    r41 = del_t * fx41 / m1
    s41 = del_t * fy41 / m1
    r42 = del_t * fx41 * (-1.0) / m2
    s42 = del_t * fy41 * (-1.0) / m2

    x1 += (k11 + 2.0 * k21 + 2.0 * k31 + k41) / 6.0
    y1 += (l11 + 2.0 * l21 + 2.0 * l31 + l41) / 6.0
    vx1 += (r11 + 2.0 * r21 + 2.0 * r31 + r41) / 6.0
    vy1 += (s11 + 2.0 * s21 + 2.0 * s31 + s41) / 6.0
    x2 += (k12 + 2.0 * k22 + 2.0 * k32 + k42) / 6.0
    y2 += (l12 + 2.0 * l22 + 2.0 * l32 + l42) / 6.0
    vx2 += (r12 + 2.0 * r22 + 2.0 * r32 + r42) / 6.0
    vy2 += (s12 + 2.0 * s22 + 2.0 * s32 + s42) / 6.0

    t_list[i+1] = t
    x1_list[i+1] = x1
    y1_list[i+1] = y1
    vx1_list[i+1] = vx1
    vy1_list[i+1] = vy1
    x2_list[i+1] = x2
    y2_list[i+1] = y2
    vx2_list[i+1] = vx2
    vy2_list[i+1] = vy2


# 間引きレートを定義
skip_steps = 10 # 10ステップごとに描画
N_frames = N // skip_steps # 新しいフレーム総数

# アニメーションのための設定
fig, ax = plt.subplots(figsize=(8.0, 6.0)) # figとaxを分けて取得
ax.set_xlabel("x")
ax.set_xlim(-25, 25)
ax.set_ylabel("y")
ax.set_ylim(-30, 30)
ax.grid(True)
ax.set_title("Two Body Gravitational Problem")


# 描画要素の初期化
# 粒子の軌跡全体を格納するLine2Dオブジェクト
line1, = ax.plot([], [], color="b", label="Trajectory1")
line2, = ax.plot([], [], color="r", label="Trajectory2")
# 現在の粒子の位置をマークするScatterオブジェクト
point1, = ax.plot([], [], 'o', color="g", markersize=8, label="Particle1 Position")
point2, = ax.plot([], [], 'o', color="y", markersize=8, label="Particle2 Position")


# アニメーションの初期化関数
def init():
    line1.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    point2.set_data([], [])
    return line1, point1, line2, point2

# アニメーションの更新関数
# i はフレーム番号
def update(i):
    # 描画するデータのインデックスを計算
    data_index = i * skip_steps

    # 軌跡は data_index までを描画
    x1_data = x1_list[:data_index+1]
    y1_data = y1_list[:data_index+1]
    x2_data = x2_list[:data_index+1]
    y2_data = y2_list[:data_index+1]

    line1.set_data(x1_data, y1_data)
    line2.set_data(x2_data, y2_data)

    # 現在の位置も data_index の点を描画
    point1.set_data([x1_list[data_index]], [y1_list[data_index]])
    point2.set_data([x2_list[data_index]], [y2_list[data_index]])


    # タイムスタンプをグラフに追加する場合は以下の行を使用
    # ax.set_title(f"Time: {t_list[data_index]:.2f} s")

    return line1, point1, line2, point2

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