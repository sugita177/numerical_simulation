import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# SciPyのODEソルバをインポート
from scipy.integrate import solve_ivp

# --------------------
# 1. 定数
# --------------------

G = 1.0
EPS = 1e-1  # 重力計算の特異点を避けるための緩和パラメータ

# --------------------
# 2. 導関数 f(t, y) の定義 (solve_ivp用)
# --------------------


def n_body_derivative(
        t: float, state_flat: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    solve_ivp が要求する形式の導関数 dy/dt = f(t, y) を計算する。
    ここで、y は状態変数 (x, y, vx, vy) の平坦化された配列。

    Args:
        t (float): 現在の時刻 (使用しないが solve_ivp のために必要)
        state_flat (np.ndarray): 平坦化された状態変数 (N_body * 4)
                                 (x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...)
        masses (np.ndarray): 各粒子の質量 (N_body,)

    Returns:
        np.ndarray: 状態変数の導関数 (d(state)/dt) (N_body * 4)
    """
    N_body = len(masses)
    # 状態変数を (N_body, 4) の形状に戻す: (x, y, vx, vy)
    state = state_flat.reshape(N_body, 4)
    positions = state[:, :2]    # (N_body, 2)
    velocities = state[:, 2:]   # (N_body, 2)

    # 加速度を格納する配列を初期化 (ax, ay)
    accelerations = np.zeros((N_body, 2))

    # NumPyのブロードキャスト機能を使って、より効率的に加速度を計算
    # R_diff[i, j] は (pos[j] - pos[i]) のベクトル
    # (N_body, N_body, 2)
    R_diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]

    # 距離の二乗 r^2 = ||R_diff||^2
    # R_squared[i, j] は i と j の間の距離の二乗
    R_squared = np.sum(R_diff**2, axis=2)  # (N_body, N_body)

    # 距離 r = sqrt(r^2 + EPS^2)
    R = np.sqrt(R_squared + EPS**2)  # (N_body, N_body)

    # 重力定数と質量項 G * mj / r^3
    # M_term[i, j] は j から i にかかる G * mj / r_ij^3 のスカラー項
    M_term = G * masses[np.newaxis, :] / (R**3)  # (N_body, N_body)

    # 自分自身との相互作用 (R[i, i] = EPS) を無視（加速度への寄与をゼロにする）
    np.fill_diagonal(M_term, 0.0)

    # 全ての粒子からの加速度を合計
    # accel[i] = Sum_j(M_term[i, j] * R_diff[i, j])
    # Einstein summation (einsum) を使用すると高速
    # 'ij, ijk -> ik': N_body * N_body と N_body * N_body * 2 を計算し、
    #                最初の N_body 軸で合計して N_body * 2 を得る
    accelerations = np.einsum('ij, ijk -> ik', M_term, R_diff)  # (N_body, 2)

    # 導関数 (dy/dt) は (d_pos/dt, d_vel/dt) = (vel, accel)
    # np.hstack で (N_body, 4) に結合し、さらに平坦化
    # (N_body * 4,)
    derivative_state = np.hstack((velocities, accelerations)).flatten()

    return derivative_state


# --------------------
# 3. シミュレーション設定と実行
# --------------------

# 時間設定
t_0: float = 0.0
t_f: float = 100.0
# solve_ivp はアダプティブステップサイズを採用するため、del_t は必要ないが、
# 出力点の制御のために t_span と t_points を定義する
del_t_output: float = 1e-2  # 結果を記録する間隔
t_span = (t_0, t_f)
t_points = np.arange(t_0, t_f + del_t_output, del_t_output)  # 出力時間点

# 粒子設定 (N_body)
N_body: int = 3

# 質量 (np.ndarray)
masses = np.array([100.0, 150.0, 100.0])

# 初期条件 (N_body, 4) の配列: (x, y, vx, vy)
initial_conditions_array = np.zeros((N_body, 4))
initial_conditions_array[0] = [-10.0, 0.0, 0.5, 4.0]
initial_conditions_array[1] = [0.0, 0.0, 0.0, 0.0]
initial_conditions_array[2] = [10.0, 0.0, -0.5, -3.99]

# solve_ivp のために初期条件を平坦化
initial_conditions_flat = initial_conditions_array.flatten()

print("シミュレーション開始...")

# solve_ivp を使用して積分を実行
# 'RK45' はデフォルトの積分器で、アダプティブステップサイズの4(5)次のルンゲ・クッタ法
# t_eval に出力時間点を指定することで、必要な点の状態を取得できる
solution = solve_ivp(
    n_body_derivative,
    t_span,
    initial_conditions_flat,
    method='RK45',  # 推奨されるアダプティブステップ法
    args=(masses,),  # 導関数に追加で渡す引数
    t_eval=t_points  # 結果を評価する時間点
)

print(f"シミュレーション完了。状態: {solution.message}")
# N_steps は solve_ivp の t_eval の長さ
N_steps = len(solution.t)

# 結果を元の形状 (N_steps, N_body, 4) に戻す
history = solution.y.T.reshape(N_steps, N_body, 4)
t_list = solution.t


# --------------------
# 4. アニメーション (変更なし)
# --------------------

# シミュレーション完了後に全範囲を固定で計算
all_x_final = history[:, :, 0].flatten()
all_y_final = history[:, :, 1].flatten()

padding = 5.0
x_min, x_max = np.min(all_x_final) - padding, np.max(all_x_final) + padding
y_min, y_max = np.min(all_y_final) - padding, np.max(all_y_final) + padding

# 間引きレートを定義
skip_steps = 10  # 10ステップごとに描画
N_frames = N_steps // skip_steps  # 新しいフレーム総数

fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.set_xlabel("x")
ax.set_xlim(x_min, x_max)
ax.set_ylabel("y")
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal', adjustable='box')  # グラフを正方形に保つ
ax.grid(True)
ax.set_title(f"N-Body Gravitational Problem (N={N_body})")

# 粒子ごとに色を設定
colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
if N_body > len(colors):
    print("Warning: N_body is greater than available colors."
          + "Colors will be repeated.")

# 描画要素の初期化
lines = []  # 軌跡
points = []  # 現在の位置
for i in range(N_body):
    color = colors[i % len(colors)]
    # 軌跡
    line, = ax.plot([], [], color=color, alpha=0.5, label=f"Trajectory{i+1}")
    lines.append(line)
    # 現在の位置
    # 質量に応じてマーカーサイズを変えても良い
    marker_size = 5 + 3 * (masses[i] > 10)
    point, = ax.plot([], [], 'o', color=color, markersize=marker_size)
    points.append(point)


# アニメーションの初期化関数
def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        point.set_data([], [])
    return lines + points


# アニメーションの更新関数
def update(i):
    data_index = i * skip_steps

    for j in range(N_body):
        # j番目の粒子の (x, y) 座標の全履歴
        x_history = history[:data_index+1, j, 0]
        y_history = history[:data_index+1, j, 1]

        # 軌跡を更新
        lines[j].set_data(x_history, y_history)

        # 現在の位置を更新
        points[j].set_data([x_history[-1]], [y_history[-1]])

    ax.set_title(f"N-Body Gravitational Problem (N={N_body})"
                 + f" - Time: {t_list[data_index]:.2f}")

    return lines + points


# アニメーションの作成
ani = FuncAnimation(
    fig,
    update,
    frames=N_frames,
    init_func=init,
    interval=10,
    blit=True
)

ax.legend()
plt.show()
