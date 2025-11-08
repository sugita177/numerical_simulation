# diffusion equation
# 1d space
# finite difference method

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初期値を返す
def get_intial_state(Nx, Q):
    u0 = np.zeros(Nx)
    u0[Nx // 2] = Q
    return u0


# 定数
t_i = 0.0
t_f = 100.0
x_min = 0.0
x_max = 10.0
del_t = 0.01
del_x = 0.2
k = 1.0 # 拡散率
coeff = k * del_t / (del_x**2) # λ (クーラン数)

# λ <= 0.5であること。
if coeff > 0.5:
    print(f"警告: クーラン数 (coeff) が0.5を超えています ({coeff})。計算が発散する可能性があります。")

Nt: int = int((t_f - t_i) / del_t)
Nx: int = int((x_max - x_min) / del_x) + 1

u_list = np.zeros([Nt+1, Nx])

# メイン処理
# 初期化
Q = 1.0
u_list[0] = get_intial_state(Nx, Q)

# 更新処理
for i in range(Nt):
    t = i * del_t

    u_list[i+1, 1:-1] = u_list[i, 1:-1] + coeff * (u_list[i, 2:] - 2 * u_list[i, 1:-1] + u_list[i, :-2])
    # 境界条件 (ディリクレ条件: 両端の温度を0に固定)
    u_list[i+1, 0] = 0.0
    u_list[i+1, -1] = 0.0

print(f"シミュレーションが完了しました。全ステップ数: {Nt}")

# アニメーションの作成
# 間引きレート
skip_steps = 1
N_frames = Nt // skip_steps

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.set_xlabel("x")
ax.set_xlim(x_min, x_max)
ax.set_ylabel("u")
ax.set_ylim(0.0, Q + 0.1)
ax.grid(True)
ax.set_title("Diffusion equation in 1 dimension")

line, = ax.plot([], [], color="r", label="Temperature")
x_data = np.linspace(x_min, x_max, Nx)

def init():
    line.set_data([], [])
    return line,

def update(i):
    data_index = i * skip_steps
    u_data = u_list[data_index]
    line.set_data(x_data, u_data)

    return line,

ani = FuncAnimation(
    fig, 
    update, 
    frames=N_frames, # 間引き後のフレーム数を使用
    init_func=init, 
    interval=20, # 指定した数値のms ごとに更新 (アニメーションの速度を調整)
    blit=True
)

ax.legend()

plt.show()
