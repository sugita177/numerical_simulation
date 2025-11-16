# wave equation
# 2d space
# finite difference method
# free boundary condition

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from mpl_toolkits.mplot3d import Axes3D

# --- パラメータの設定（変更なし） ---
t_i = 0.0
t_f = 100.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0
del_t = 0.01
del_x = 0.05
del_y = del_x
v = 1.0  # 波の速さ

C = v * del_t / del_x
coeff = C**2  # 更新式で使用する C^2

# 2次元のCFL条件チェック
if C > (1.0 / np.sqrt(2.0)):
    print(f"警告: 2次元クーラン数 C が 1/sqrt(2) を超えています ({C:.4f})。\
          計算が発散する可能性があります。推奨: C <= 0.707")

Nt: int = int((t_f - t_i) / del_t)
# --- 修正点 1: 物理的な格子点数 ---
Nx_physical: int = int((x_max - x_min) / del_x) + 1
Ny_physical: int = int((y_max - y_min) / del_y) + 1

# --- 修正点 2: 仮想点を含めた全格子点数 (N_physical + 2) ---
Nx_total = Nx_physical + 2
Ny_total = Ny_physical + 2

# u_listの形状を (Nt+1, Nx_total, Ny_total) に変更
u_list = np.zeros([Nt+1, Nx_total, Ny_total])
# x_data, y_dataは物理的な領域のみで作成
x_data = np.linspace(x_min, x_max, Nx_physical)
y_data = np.linspace(y_min, y_max, Ny_physical)


# 初期値を返す関数 (物理領域のサイズに合わせて調整)
def get_initial_state_2d_gaussian(x_data, y_data, U, sigma_ratio=0.1):
    Nx, Ny = x_data.size, y_data.size
    # u0は物理領域のサイズ (Nx x Ny) で作成
    u0 = np.zeros((Nx, Ny))
    x_center, y_center = \
        (x_data[-1] - x_data[0]) / 2.0, (y_data[-1] - y_data[0]) / 2.0
    L = x_data[-1] - x_data[0]
    sigma = L * sigma_ratio

    # 2次元座標メッシュを作成
    X, Y = np.meshgrid(x_data, y_data, indexing='ij')

    # 2次元ガウス波形 (xとyからの距離に基づいて振幅を設定)
    distance_sq = (X - x_center)**2 + (Y - y_center)**2
    u0 = U * np.exp(-distance_sq / (2.0 * sigma**2))

    return u0


# メイン処理
# 初期化
U = 1.0
sigma_ratio = 0.05
initial_u_physical\
    = get_initial_state_2d_gaussian(x_data, y_data, U, sigma_ratio)

# 初期値を物理領域 (インデックス 1:-1) にセット
u_list[0, 1:-1, 1:-1] = initial_u_physical
u_list[1, 1:-1, 1:-1] = initial_u_physical

# --- 自由端の初期条件（仮想点の設定） ---
# t=0, t=1の仮想点も隣接点と同じ値にする
for t in range(2):
    # X方向の仮想点 (インデックス 0 と -1)
    u_list[t, 0, :] = u_list[t, 1, :]
    u_list[t, -1, :] = u_list[t, -2, :]
    # Y方向の仮想点 (インデックス 0 と -1)
    u_list[t, :, 0] = u_list[t, :, 1]
    u_list[t, :, -1] = u_list[t, :, -2]


# 更新処理
for i in range(1, Nt):
    # --- 修正点 3: 自由端反射境界条件 (仮想点の設定) ---
    # 時刻 i の状態を使って、時刻 i+1 の計算を行う前に仮想点を設定する

    # X方向の仮想点
    # x=x_min側 (インデックス 0) の仮想点に、内側の隣接点 (インデックス 1) の値をコピー
    u_list[i, 0, :] = u_list[i, 1, :]
    # x=x_max側 (インデックス -1) の仮想点に、内側の隣接点 (インデックス -2) の値をコピー
    u_list[i, -1, :] = u_list[i, -2, :]

    # Y方向の仮想点
    # y=y_min側 (インデックス 0) の仮想点に、内側の隣接点 (インデックス 1) の値をコピー
    u_list[i, :, 0] = u_list[i, :, 1]
    # y=y_max側 (インデックス -1) の仮想点に、内側の隣接点 (インデックス -2) の値をコピー
    u_list[i, :, -1] = u_list[i, :, -2]

    # ----------------------------------------------------

    # 2次元ラプラシアン (5点差分) を計算
    # --- 修正点 4: 適用範囲を全物理領域 (インデックス 1:-1) に拡大 ---
    # 境界点 (インデックス 1, -2) も含めて計算
    laplacian_u = (
        # x方向の隣接点 (仮想点も使用)
        u_list[i, 2:, 1:-1] + u_list[i, :-2, 1:-1] +
        # y方向の隣接点 (仮想点も使用)
        u_list[i, 1:-1, 2:] + u_list[i, 1:-1, :-2] -
        # 中央の点
        4 * u_list[i, 1:-1, 1:-1]
        )

    # 更新式 (物理領域全体に適用)
    u_list[i+1, 1:-1, 1:-1] = \
        2.0 * u_list[i, 1:-1, 1:-1] - u_list[i-1, 1:-1, 1:-1]\
        + coeff * laplacian_u

    # Note: 時刻 i+1 の仮想点の設定は、次のループの開始時に行われるため、
    # ここでは境界条件のコードは不要です。

    print(f"{i}ステップ完了")

print(f"シミュレーションが完了しました。全ステップ数: {Nt-1}")

# --- 3Dアニメーションの作成 ---
skip_steps = 5
N_frames = Nt // skip_steps

fig = plt.figure(figsize=(10, 8))
ax: Axes3D = fig.add_subplot(111, projection='3d')  # type: ignore

# 物理的なプロット用のメッシュグリッドを作成
# u_list[..., 1:-1, 1:-1] が物理領域に対応
X, Y = np.meshgrid(x_data, y_data, indexing='ij')

# プロットの初期設定
U_abs = math.fabs(U)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("U")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(-U_abs, U_abs)
ax.set_title("2D Wave Equation Simulation (3D Surface - Free Boundary)")

# 最初のサーフェスプロットを作成
surface = ax.plot_surface(
            X, Y, u_list[0, 1:-1, 1:-1], cmap='viridis', rstride=1, cstride=1,
            vmin=-U_abs, vmax=U_abs
        )


def update(i):
    global surface

    data_index = i * skip_steps
    # --- 修正点 5: 物理領域のみをプロット ---
    current_u_physical = u_list[data_index, 1:-1, 1:-1]

    surface.remove()

    surface = ax.plot_surface(
        X, Y, current_u_physical, cmap='viridis', rstride=1, cstride=1,
        vmin=-U_abs, vmax=U_abs
        )

    t_current = data_index * del_t
    ax.set_title(f"2D Wave Equation (Time: {t_current:.2f}, Free Boundary)")

    return [surface]


ani = FuncAnimation(
    fig,
    update,
    frames=N_frames,
    interval=100,
    blit=False
)

plt.show()
