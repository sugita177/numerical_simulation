# Monte Carlo Method
# Ising Model in 2 dimension

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import math


def calc_energy(s, J, H):
    # 周期境界条件

    # 1. 右隣との相互作用 (axis=1 で右に1つシフト)
    interaction_right = np.roll(s, shift=1, axis=1)

    # 2. 左隣との相互作用 (axis=1 で左に1つシフト)
    interaction_left = np.roll(s, shift=-1, axis=1)

    # 3. 下隣との相互作用 (axis=0 で下に1つシフト)
    interaction_down = np.roll(s, shift=1, axis=0)

    # 4. 上隣との相互作用 (axis=0 で上に1つシフト)
    interaction_up = np.roll(s, shift=-1, axis=0)

    # 全隣接スピンの合計
    Neighbor_Sum\
        = interaction_right + interaction_left\
        + interaction_down + interaction_up

    # 全エネルギーの相互作用項を計算
    # E = -J * (1/2) * sum(S_i * Neighbor_Sum_i)
    # 1/2 をかけるのは、各相互作用 (S_i S_j) を 2 回（S_i側とS_j側）数えているため。
    E_interaction_sum = -J * np.sum(s * Neighbor_Sum) / 2.0
    E_mag = -1.0 * np.sum(s) * H

    # 全エネルギー (外部磁場 H=0 の場合)
    E = E_interaction_sum + E_mag
    return E


def calculate_delta_E(i, j, s, J, H, L):
    """
    格子点(i, j)のスピンを反転させたときのエネルギー変化 dE を計算する
    """
    # 周期境界条件を使って隣接スピンのインデックスを取得
    # 上隣 (i-1, j)
    s_up = s[(i - 1) % L, j]
    # 下隣 (i+1, j)
    s_down = s[(i + 1) % L, j]
    # 左隣 (i, j-1)
    s_left = s[i, (j - 1) % L]
    # 右隣 (i, j+1)
    s_right = s[i, (j + 1) % L]

    # 隣接スピンの合計 (Sum of Neighbors)
    neighbor_sum = s_up + s_down + s_left + s_right

    # デルタEの計算
    # dE = 2 * S_i,j * (J * Sum_neighbor + H)
    delta_E = 2 * s[i, j] * (J * neighbor_sum + H)

    return delta_E


# メトロポリス法で更新する
def do_metropolis_method(i, j, s, T, J, H, L):
    """
    格子点(i, j)のスピンをメトロポリス法で更新する関数。
    T: 温度
    """
    # スピン反転によるエネルギー変化を計算
    delta_E = calculate_delta_E(i, j, s, J, H, L)

    # 1. エネルギーが非増加の場合 (ΔE <= 0)：必ず反転を受容
    if delta_E <= 0.0:
        # スピンを反転させる (s[i, j]を -s[i, j]にする)
        s[i, j] *= -1
        # 更新されたスピンの値を返す（あるいは s[i, j] を直接更新する）
        return s[i, j]

    # 2. エネルギーが増加する場合 (ΔE > 0)：確率exp(-ΔE / T)で反転を受容
    else:
        # P_accept = exp(-ΔE / T) を計算
        acceptance_prob = math.exp(-delta_E / T)

        # 0から1の一様乱数と比較
        if np.random.rand() < acceptance_prob:
            # 確率的に受容された場合、スピンを反転させる
            s[i, j] *= -1
            return s[i, j]
        else:
            # 受容されなかった場合、スピンはそのまま
            return s[i, j]


def update_one_spin(s, T, J, H, L):
    # 行インデックス i をランダムに選択
    # 0 (含む) から L (含まない) の範囲で整数を生成
    i = np.random.randint(0, L)

    # 列インデックス j をランダムに選択
    # 0 (含む) から L (含まない) の範囲で整数を生成
    j = np.random.randint(0, L)

    # これで、ランダムに選ばれたスピンの位置は s[i, j] となる
    s[i, j] = do_metropolis_method(i, j, s, T, J, H, L)

    return s


# --- シミュレーションパラメータ ---
L = 20
N = L**2
s = np.random.choice([-1, 1], size=(L, L))
J = 1.0  # スピン相互作用係数
T = 1.0  # 温度 (k_B=1)
H = 0.0  # 外部磁場

# --- メインシミュレーション設定 ---
N_MC_STEPS = 5000  # モンテカルロステップの総回数
N_THERM = 1000  # 熱平衡化ステップ数
N_MEASURE = N_MC_STEPS - N_THERM  # 測定ステップ数

# 観測量（エネルギーと磁化）を記録するリスト
energy_history = []
magnetization_history = []

# モンテカルロシミュレーションの実行
print(f"--- Simulating Ising Model (L={L}, T={T}, H={H}) ---")

# 熱平衡化フェーズ
for step in range(N_THERM):
    # N個のスピンをランダムに選んで更新 (1 MCステップ)
    for _ in range(N):
        s = update_one_spin(s, T, J, H, L)

    if (step + 1) % 500 == 0:
        print(f"Thermalizing: Step {step + 1}/{N_THERM}")

# 測定フェーズ
for step in range(N_MEASURE):
    # N個のスピンをランダムに選んで更新 (1 MCステップ)
    for _ in range(N):
        s = update_one_spin(s, T, J, H, L)

    # 観測量の計算と記録
    current_E = calc_energy(s, J, H)
    current_M = np.sum(s)

    energy_history.append(current_E / N)  # 1スピンあたりのエネルギー
    magnetization_history.append(current_M / N)  # 1スピンあたりの磁化

    if (step + 1) % 500 == 0:
        print(f"Measuring: Step {step + 1}/{N_MEASURE}, "
              + f"E/N={current_E/N:.4f}, M/N={current_M/N:.4f}")


# 結果の表示
print("\n--- Simulation Results ---")

# 平均値の計算
avg_E = np.mean(energy_history)
avg_M = np.mean(magnetization_history)

print(f"Average Energy per spin (E/N): {avg_E:.6f}")
print(f"Average Magnetization per spin (|M|/N): {abs(avg_M):.6f}")

# 時系列プロット
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(energy_history)
plt.title(f'Energy per Spin (T={T})')
plt.xlabel('MC Steps')
plt.ylabel('E/N')

plt.subplot(1, 2, 2)
plt.plot(magnetization_history)
plt.title(f'Magnetization per Spin (T={T})')
plt.xlabel('MC Steps')
plt.ylabel('M/N')

plt.tight_layout()
plt.show()

# 最終スピン配置の表示
plt.figure(figsize=(5, 5))
plt.imshow(s, cmap='gray')
plt.title(f'Final Spin Configuration (T={T})')
plt.show()
