# wave equation
# 1d space
# finite difference method

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初期値を返す
def get_intial_state_sin(x_data, U, wave_width_ratio=0.2):
    """
    中央部にサインカーブを持つ初期状態を返す
    """
    Nx = x_data.size
    u0 = np.zeros(Nx)
    x_center = (x_data[0] + x_data[-1]) / 2.0
    L = x_data[-1] - x_data[0]

    # 波を配置する範囲 (例: 全長の 20%)
    wave_width = L * wave_width_ratio
    x_start = x_center - wave_width / 2.0
    x_end = x_center + wave_width / 2.0

    # 該当するインデックスを特定
    idx = np.where((x_data >= x_start) & (x_data <= x_end))

    # サイン波の生成
    # 0 から pi まで変化する引数を作成し、波を中央に配置
    arg = np.pi * (x_data[idx] - x_start) / wave_width
    u0[idx] = U * np.sin(arg)

    return u0
# -----------------------------------


# 定数
t_i = 0.0
t_f = 100.0
x_min = 0.0
x_max = 10.0
del_t = 0.01
del_x = 0.05
v = 0.5 # 波の速さ
coeff = (v * del_t / del_x)**2

# クーラン条件 (C <= 1) の確認 (ここでは C^2 <= 1)
if coeff > 1.0:
    print(f"警告: クーラン数 C が1を超えています ({np.sqrt(coeff):.2f})。計算が発散する可能性があります。")

Nt: int = int((t_f - t_i) / del_t)
Nx: int = int((x_max - x_min) / del_x) + 1

u_list = np.zeros([Nt+1, Nx])
x_data = np.linspace(x_min, x_max, Nx) # x座標のデータを作成

# メイン処理
# 初期化
U = 1.0
u_list[0] = get_intial_state_sin(x_data, U)
u_list[1] = get_intial_state_sin(x_data, U)

# 更新処理
for i in range(1, Nt):
    t = i * del_t

    u_list[i+1, 1:-1] = 2.0 * u_list[i, 1:-1] - u_list[i-1, 1:-1] + coeff * (u_list[i, 2:] - 2 * u_list[i, 1:-1] + u_list[i, :-2])
    # 境界条件 (ディリクレ条件: 両端の温度を0に固定)
    u_list[i+1, 0] = 0.0
    u_list[i+1, -1] = 0.0

print(f"シミュレーションが完了しました。全ステップ数: {Nt}")

# アニメーションの作成
# 間引きレート
skip_steps = 5
N_frames = Nt // skip_steps

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.set_xlabel("x")
ax.set_xlim(x_min, x_max)
ax.set_ylabel("u")
ax.set_ylim(-U - 0.1, U + 0.1)
ax.grid(True)
ax.set_title("Wave equation in 1 Dimension")

line, = ax.plot([], [], color="b", label="Amplitude")
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
