# diffusion equation
# 2d space
# finite difference method

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# 初期値を返す
def get_initial_state_2d_gaussian(x_data, y_data, U, sigma_ratio=0.1):
    Nx, Ny = x_data.size, y_data.size
    u0 = np.zeros((Nx, Ny))
    x_center, y_center = (x_data[-1] - x_data[0]) / 2.0, (y_data[-1] - y_data[0]) / 2.0
    L = x_data[-1] - x_data[0]
    sigma = L * sigma_ratio

    # 2次元座標メッシュを作成
    X, Y = np.meshgrid(x_data, y_data, indexing='ij') # 'ij'インデックスで Nx x Ny の形状にする

    # 2次元ガウス波形 (xとyからの距離に基づいて振幅を設定)
    distance_sq = (X - x_center)**2 + (Y - y_center)**2
    u0 = U * np.exp(-distance_sq / (2.0 * sigma**2))
    
    return u0
    

# 定数
t_i = 0.0
t_f = 10.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0
del_t = 0.005
del_x = 0.1
del_y = del_x
k = 0.1 # 拡散率
coeff = k * del_t / (del_x**2) # λ (クーラン数)

# 2D 5点差分陽解法での安定性条件は coeff <= 0.25 であるべき
if coeff > 0.25:
    # 警告メッセージも修正
    print(f"**重大な警告**: クーラン数 (coeff) が0.25を超えています ({coeff})。計算が発散します。")

Nt: int = int((t_f - t_i) / del_t)
Nx: int = int((x_max - x_min) / del_x) + 1
Ny: int = int((y_max - y_min) / del_y) + 1

u_list = np.zeros([Nt+1, Nx, Ny])
x_data = np.linspace(x_min, x_max, Nx)
y_data = np.linspace(y_min, y_max, Ny)

# メイン処理
# 初期化
U = 2.0
sigma_ratio = 0.15
u_list[0] = get_initial_state_2d_gaussian(x_data, y_data, U, sigma_ratio)

# 更新処理
for i in range(0, Nt):
    # 2次元ラプラシアン (5点差分) を計算
    laplacian_u = (u_list[i, 2:, 1:-1] + u_list[i, :-2, 1:-1] +  # x方向の隣接点
                   u_list[i, 1:-1, 2:] + u_list[i, 1:-1, :-2] -  # y方向の隣接点
                   4 * u_list[i, 1:-1, 1:-1])                    # 中央の点

    # 更新式
    u_list[i+1, 1:-1, 1:-1] = u_list[i, 1:-1, 1:-1] + coeff * laplacian_u
    
    # 境界条件 (ここでは固定端(u=0)と仮定)
    u_list[i+1, 0, :] = 0.0     # x=x_min (左境界)
    u_list[i+1, -1, :] = 0.0    # x=x_max (右境界)
    u_list[i+1, :, 0] = 0.0     # y=y_min (下境界)
    u_list[i+1, :, -1] = 0.0    # y=y_max (上境界)

    # print(f"{i}ステップ完了")

print(f"シミュレーションが完了しました。全ステップ数: {Nt}")

# --- 3Dアニメーションの作成 ---
skip_steps = 5 # 3Dプロットは重いので、間引きレートを増やす
N_frames = Nt // skip_steps

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d') # 3Dプロットを有効にする

# 3Dサーフェスプロットのためのメッシュグリッドを作成
X, Y = np.meshgrid(x_data, y_data, indexing='ij')

# プロットの初期設定
U_abs = math.fabs(U)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("U")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(-U_abs, U_abs) # Z軸の範囲も設定
ax.set_title("2D Diffustion Equation Simulation (3D Surface)")

# 最初のサーフェスプロットを作成し、それを更新していく
# rstride, cstride はグリッドの間引き。視覚的にスムーズにするため、ある程度大きな値が良い
surface = ax.plot_surface(X, Y, u_list[0], cmap='magma', rstride=1, cstride=1, vmin=-U_abs, vmax=U_abs)

def update(i):
    global surface # グローバル変数としてsurfaceを参照できるようにする（推奨はしないが、手っ取り早い）
    
    data_index = i * skip_steps
    current_u = u_list[data_index] # 2次元配列のままにする
    
    # 既存のサーフェスを削除
    surface.remove()
    
    # 新しいデータでサーフェスを再作成
    # rstride, cstride はグリッドの間引き。視覚的にスムーズにするため、ある程度大きな値が良い
    surface = ax.plot_surface(X, Y, current_u, cmap='magma', rstride=1, cstride=1, vmin=-U_abs, vmax=U_abs)
    
    # タイトルのみ更新
    t_current = data_index * del_t
    ax.set_title(f"2D Diffustion Equation (Time: {t_current:.2f})")
    
    return [surface] # surface オブジェクトを返す

ani = FuncAnimation(
    fig, 
    update, 
    frames=N_frames, 
    interval=100, # アニメーションの速度
    blit=False # 3Dプロットでは blit=True は非推奨/動作しないことが多い
)

plt.show()