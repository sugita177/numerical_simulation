import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------
# 1. ヘルパー関数
# --------------------

# eps softening parameter (重力計算の特異点を避けるための緩和パラメータ)
G = 1.0
EPS = 1e-1


# 粒子の加速度を計算する関数
def calculate_acceleration(
        masses: np.ndarray, positions: np.ndarray
        ) -> np.ndarray:
    """
    全ての粒子の相互作用による加速度を計算する。

    Args:
        masses (np.ndarray): 各粒子の質量 (N_body,)
        positions (np.ndarray): 各粒子の位置 (N_body, 2) (x, y)

    Returns:
        np.ndarray: 各粒子の加速度 (N_body, 2) (ax, ay)
    """
    N_body = len(masses)
    # 加速度を格納する配列を初期化 (ax, ay)
    accelerations = np.zeros((N_body, 2))

    # 各粒子 i について、他の全ての粒子 j からの重力を合計
    for i in range(N_body):
        for j in range(N_body):
            # i != j の場合のみ計算
            if i != j:
                # 粒子 i から j への相対位置ベクトル (rx, ry)
                # r_vector = pos_j - pos_i
                r_vector = positions[j] - positions[i]

                # 距離の二乗: r^2 = (xj - xi)^2 + (yj - yi)^2
                r_squared = np.sum(r_vector**2)

                # 距離: r
                r = np.sqrt(r_squared + EPS**2)  # ソフトニング込みの距離

                # 粒子 i にかかる、粒子 j からの力 F_ij = G * mi * mj / r^2 * (r_vector / r)
                # 粒子 i にかかる加速度 a_i = F_ij / m_i = G * m_j / r^3 * r_vector

                # スカラー加速度の絶対値 |a_i| = G * m_j / (r_squared + EPS**2)
                accel_abs = G * masses[j] / (r**3)

                # 加速度ベクトル (ax, ay) = accel_abs * r_vector
                accel_vector = accel_abs * r_vector

                # 総力（総加速度）に加算
                accelerations[i] += accel_vector

    return accelerations


# --------------------
# 2. ルンゲ・クッタ (RK4) のステップ関数
# --------------------

def runge_kutta_step(
        del_t: float, masses: np.ndarray, current_state: np.ndarray
        ):
    """
    ルンゲ・クッタ法（4次）で1ステップ時間の積分を行う。

    Args:
        del_t (float): タイムステップ Delta t
        masses (np.ndarray): 粒子の質量 (N_body,)
        current_state (np.ndarray): 現在の状態変数 (N_body, 4) (x, y, vx, vy)

    Returns:
        np.ndarray: 次の時間の状態変数 (N_body, 4)
    """
    # 状態変数の分解
    positions = current_state[:, :2]  # (x, y)
    velocities = current_state[:, 2:]  # (vx, vy)

    # 状態ベクトルの導関数 f(y) = (v, a) の計算
    def derivative(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        # pos: (N_body, 2), vel: (N_body, 2)
        accelerations = calculate_acceleration(masses, pos)  # (N_body, 2)
        # 導関数 (dy/dt) は (d_pos/dt, d_vel/dt) = (vel, accel)
        # np.hstack で (N_body, 4) に結合
        return np.hstack((vel, accelerations))

    # K1
    deriv1 = derivative(positions, velocities)
    k1 = del_t * deriv1

    # K2
    state2 = current_state + k1 / 2.0
    deriv2 = derivative(state2[:, :2], state2[:, 2:])
    k2 = del_t * deriv2

    # K3
    state3 = current_state + k2 / 2.0
    deriv3 = derivative(state3[:, :2], state3[:, 2:])
    k3 = del_t * deriv3

    # K4
    state4 = current_state + k3
    deriv4 = derivative(state4[:, :2], state4[:, 2:])
    k4 = del_t * deriv4

    # 次の状態を計算
    next_state = current_state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    return next_state


# --------------------
# 3. シミュレーション設定と実行
# --------------------

# 時間設定
t_0: float = 0.0
t_f: float = 100.0
del_t: float = 1e-2
N_steps: int = round((t_f - t_0) / del_t)

# 粒子設定 (N_body)
N_body: int = 3  # 粒子の数

# 質量 (np.ndarray)
masses = np.array([100.0, 150.0, 100.0])

# 初期条件 (N_body, 4) の配列として定義: (x, y, vx, vy)
initial_conditions = np.zeros((N_body, 4))

# 粒子1 (x, y, vx, vy)
initial_conditions[0] = [-10.0, 0.0, 0.5, 4.0]
# 粒子2 (x, y, vx, vy)
initial_conditions[1] = [0.0, 0.0, 0.0, 0.0]
# 粒子3 (x, y, vx, vy)
initial_conditions[2] = [10.0, 0.0, -0.5, -3.99]

# 結果格納用の配列 (N_steps, N_body, 4)
# (時間ステップ, 粒子番号, 状態変数(x, y, vx, vy))
history = np.zeros((N_steps, N_body, 4))
history[0] = initial_conditions

# 現在の状態変数 (N_body, 4)
current_state = initial_conditions.copy()

# シミュレーション実行
t_list = np.zeros(N_steps)
t_list[0] = t_0

for i in range(N_steps - 1):
    # RK4 ステップの実行
    current_state = runge_kutta_step(del_t, masses, current_state)

    # 結果の格納
    history[i+1] = current_state
    t_list[i+1] = t_list[i] + del_t


# --------------------
# 4. アニメーション
# --------------------

print(f"シミュレーション完了。ステップ数: {N_steps}")

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
ax.set_xlim(x_min, x_max)  # 描画範囲を適宜調整
ax.set_ylabel("y")
ax.set_ylim(y_min, y_max)  # 描画範囲を適宜調整
ax.grid(True)
ax.set_title(f"N-Body Gravitational Problem (N={N_body})")

# 粒子ごとに色を設定
colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']  # 7体まで対応
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

# アニメーションを保存したい場合は以下のコメントを外す (要: ffmpeg または pillow)
# ani.save('n_body_problem.gif', writer='pillow', fps=60)
