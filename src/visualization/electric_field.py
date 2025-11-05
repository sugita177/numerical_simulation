# visualization of an electric field

import numpy as np
import matplotlib.pyplot as plt

# 電荷クラス
class Charge:
    def __init__(self, q, x, y, z=0.0) -> None:
        self.q = q
        self.x = x
        self.y = y
        self.z = z

# 電荷の設定
def initialize_charges():
    N = 2 # 電荷数
    charges = np.ndarray([
        Charge(1.0, -5.0, 0.0),
        Charge(-1.0, 5.0, 0.0, 0.0),
    ])
    return charges

def get_charge_property_arrays(charges):
    N = charges.size
    x_list = np.ndarray(N)
    y_list = np.ndarray(N)
    z_list = np.ndarray(N)
    q_list = np.ndarray(N)
    for i in range(N):
        x_list[i] = charges.x
        y_list[i] = charges.y
        z_list[i] = charges.z
        q_list[i] = charges.q
    return q_list, x_list, y_list, z_list

# メイン処理
#クーロン定数
k = 1.0
# 電荷の設定
charges = initialize_charges()
q_list, qx_list, qy_list, _ = get_charge_property_arrays(charges)
# 測定点
Lx = 15
Ly = 15
step = 1.0

# 描画するグリッドポイントの座標を作成
# -L から L までの範囲で step 刻み
X, Y = np.meshgrid(np.arange(-Lx, Lx + step, step),
                   np.arange(-Ly, Ly + step, step))

# 電場データを格納する配列
# X, Y と同じ形状にする
Ex = np.zeros_like(X, dtype=float)
Ey = np.zeros_like(Y, dtype=float)

# 電場の計算
for i, xp in enumerate(np.arrange(-Lx, Lx + step, step)):
    for j, yp in enumerate(np.arrange(-Ly, Ly + step, step)):
        rx = xp - qx_list
        ry = yp - qy_list
        r = np.sqrt(rx**2 + ry**2)

        # 距離 r がゼロになる場合（観測点が電荷と同じ位置にある場合）のゼロ割を防ぐ
        # 極めて近い距離では電場は発散するため、ここでは計算結果をゼロにする
        with np.errstate(divide='ignore', invalid='ignore'):
            r_cubed_inv_list = np.power(r, -3.0)
            # r=0 の場合、r_cubed_inv は inf になるため、ゼロに置換
            r_cubed_inv_list[r == 0] = 0.0

        E_coeff = k * q_list * r_cubed_inv_list
        Ex_p = np.sum(E_coeff * rx)
        Ey_p = np.sum(E_coeff * ry)
        Ex[j, i] = Ex_p
        Ey[j, i] = Ey_p

# 可視化
# 電場の強さの最大値を求め、矢印の長さを正規化（見やすくするため）
E_mag = np.sqrt(Ex**2 + Ey**2)
# E_mag の最大値がゼロでない場合のみ正規化
if np.max(E_mag) > 0:
    Ex_norm = Ex / E_mag
    Ey_norm = Ey / E_mag
else:
    Ex_norm = Ex
    Ey_norm = Ey

# プロットの作成
plt.figure(figsize=(8, 8))

# 密度（skip）を調整して、矢印が重なりすぎないようにします。
skip = 2 # 2点に1点を描画
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           Ex_norm[::skip, ::skip], Ey_norm[::skip, ::skip],
           E_mag[::skip, ::skip], # 色は電場の強さ
           cmap='viridis',
           pivot='middle',
           scale=30, # 矢印の長さを調整 (小さいほど長い)
           headwidth=5, headlength=5)

# 正電荷は赤、負電荷は青で表示
# 電荷の大きさは、電荷の絶対値に比例させます（見栄えを考慮して乗数を使用）
for q, qx, qy in zip(q_list, qx_list, qy_list):
    color = 'red' if q > 0 else 'blue'
    # 電荷の絶対値が大きいほど、マーカーのサイズも大きくする
    size = abs(q) * 200 
    plt.scatter(qx, qy, s=size, c=color, marker='o', edgecolors='black', zorder=5, label='Charge' if 'Charge' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Electric Field Visualization")
plt.xlabel("x-position")
plt.ylabel("y-position")
plt.gca().set_aspect('equal', adjustable='box') # x, y 軸のスケールを合わせる
plt.xlim(-Lx - 1, Lx + 1)
plt.ylim(-Ly - 1, Ly + 1)
plt.grid(True)
plt.colorbar(label='Electric Field Magnitude (|E|)') # カラーバーを追加
plt.show()