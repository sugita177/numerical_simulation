# random_walk.py をもとにGeminiを用いて作成

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from scipy.stats import norm

# --- 1. シミュレーション設定とデータ収集 ---
M = 2000  # 試行回数 (統計的な精度のため、多めに設定)
N = 1000  # ステップ数 (計算時間短縮のため、元の10000から変更)
step_length = 1.0 # 1ステップあたりの移動距離 (元のシミュレーションより)

# データを保存するための配列
final_positions = np.zeros((M, 2))  # 最終位置 (x, y)
squared_displacements = np.zeros((M, N))  # 各試行の各ステップでの r^2

print(f"シミュレーション開始: 試行回数 M={M}, ステップ数 N={N}")

for m in range(M):
    x_list = np.zeros(N)
    y_list = np.zeros(N)
    
    squared_displacements[m, 0] = x_list[0]**2 + y_list[0]**2

    for i in range(N - 1):
        x = x_list[i]
        y = y_list[i]
        angle = random.random() * 2 * math.pi
        
        # 一定の歩幅 (step_length) で移動
        x += step_length * np.cos(angle)
        y += step_length * np.sin(angle)
        
        x_list[i+1] = x
        y_list[i+1] = y
        
        # ステップ i+1 での二乗変位を記録
        squared_displacements[m, i+1] = x_list[i+1]**2 + y_list[i+1]**2
    
    # 最終位置を記録
    final_positions[m, 0] = x_list[N-1]
    final_positions[m, 1] = y_list[N-1]

# --- 2. 統計解析 ---

# MSDを計算 (試行回数Mで平均)
msd = np.mean(squared_displacements, axis=0)
steps = np.arange(N)

# 最終位置の分散を計算
variance_x = np.var(final_positions[:, 0])
variance_y = np.var(final_positions[:, 1])

# --- 3. グラフの作成と一括表示 ---

# 2x2 のサブプロットを作成
# グラフの配置は、(1,1), (1,2), (2,1), (2,2) の4箇所
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
plt.suptitle(f'2D Random Walk Statistical Analysis (M={M} trials, N={N} steps)', fontsize=18)

# --- (1, 1) MSDのプロット (両対数) ---
ax0 = axs[0, 0]
ax0.loglog(steps[1:], msd[1:], color='blue', label='Simulated MSD')
# 理論的な線 (MSD = step_length^2 * n = 1 * n) をプロット
ax0.loglog(steps[1:], steps[1:] * step_length**2, 'k--', label='Theoretical $\propto n$')
ax0.set_xlabel('Steps ($n$)')
ax0.set_ylabel('MSD $<R^2(n)>$')
ax0.set_title('A) Mean Squared Displacement (Log-Log)')
ax0.legend()
ax0.grid(True, which="both", ls="--", alpha=0.6)

# --- (1, 2) 最終位置の2次元散布図 ---
ax1 = axs[0, 1]
ax1.scatter(final_positions[:, 0], final_positions[:, 1], s=1, alpha=0.5, color='darkgreen')
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.axvline(0, color='gray', linewidth=0.5)
ax1.set_xlabel('Final X Position ($x_N$)')
ax1.set_ylabel('Final Y Position ($y_N$)')
# 視覚的な円形分布を強調するため、スケールを統一
limit = max(abs(final_positions).max(axis=0)) * 1.1
ax1.set_xlim(-limit, limit)
ax1.set_ylim(-limit, limit)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('B) Final Positions Scatter Plot')

# --- (2, 1) X座標のヒストグラム ---
ax2 = axs[1, 0]
n_bins = int(np.sqrt(M)) # ビン数の目安
ax2.hist(final_positions[:, 0], bins=n_bins, density=True, alpha=0.6, color='skyblue', label='X positions')
ax2.set_xlabel('Final X Position ($x_N$)')
ax2.set_ylabel('Probability Density')
ax2.set_title(f'C) Final X Position Distribution ($\sigma^2$={variance_x:.2f})')

# 理論的なガウス分布をオーバーレイ
xmin, xmax = ax2.get_xlim()
x_pdf = np.linspace(xmin, xmax, 100)
p = norm.pdf(x_pdf, 0, np.sqrt(variance_x))
ax2.plot(x_pdf, p, 'r--', linewidth=2, label='Fitted Gaussian')
ax2.legend()
ax2.grid(axis='y', alpha=0.5)


# --- (2, 2) Y座標のヒストグラム ---
ax3 = axs[1, 1]
ax3.hist(final_positions[:, 1], bins=n_bins, density=True, alpha=0.6, color='lightcoral', label='Y positions')
ax3.set_xlabel('Final Y Position ($y_N$)')
ax3.set_ylabel('Probability Density')
ax3.set_title(f'D) Final Y Position Distribution ($\sigma^2$={variance_y:.2f})')

# 理論的なガウス分布をオーバーレイ
p = norm.pdf(x_pdf, 0, np.sqrt(variance_y))
ax3.plot(x_pdf, p, 'r--', linewidth=2, label='Fitted Gaussian')
ax3.legend()
ax3.grid(axis='y', alpha=0.5)

# グラフが重ならないようにレイアウトを調整
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()