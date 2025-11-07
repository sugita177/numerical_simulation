# logistic map

import numpy as np
import matplotlib.pyplot as plt

def logistic_map(x, a):
    return a * x * (1.0 - x)

# --- パラメータ設定 ---
Nx = 300         # 反復回数
x0 = 0.1         # 初期値
a_s = 2.5        # パラメータ a の開始値 (カオスが見えやすいように 0.1 から 2.5 に変更)
a_f = 4.0        # パラメータ a の終了値
step = 0.005     # パラメータ a の刻み幅 (分岐図らしくするために細かく変更)

# --- 配列の準備 ---
# Na の計算を修正
Na = int((a_f - a_s) / step) + 1 # +1 で終了値 a_f を含む
# np.zeros で初期化し、サイズを (Na, Nx) に修正
x_list = np.zeros((Na, Nx))
# np.linspace を使って a_list を作成するのが簡単
a_list = np.linspace(a_s, a_f, Na)

# --- ロジスティック写像の計算 ---
for i_a in range(Na):
    a = a_list[i_a]
    x = x0
    x_list[i_a][0] = x0
    for i in range(1, Nx):
        x = logistic_map(x, a)
        x_list[i_a][i] = x

# --- 可視化：分岐図と時系列プロットの統合 ---
N_transient = 100  # 過渡期として無視する反復回数
N_plot = Nx - N_transient 

# 描画する a の値 (収束、周期2、カオス)
a_values_to_examine = [2.8, 3.4, 3.9]
num_iterations_plot = 50 # 時系列プロットで表示する反復回数

# Figureの作成。縦長にし、複数のグラフを配置
plt.figure(figsize=(12, 10))
plt.suptitle('Logistic Map Analysis', fontsize=16)

# --- 1. 分岐図 (上の行全体 1, 2 に配置) ---
ax1 = plt.subplot(2, 1, 1) # 2行1列の1番目 (上の行全体)

for i_a in range(Na):
    a_val = a_list[i_a]
    
    # 過渡期後のxの値を取得
    x_values_to_plot = x_list[i_a][N_transient:]
    
    # 'a_val' の数だけ x の値のリストを作成
    a_values_for_scatter = np.full(N_plot, a_val)
    
    # 散布図としてプロット 
    ax1.scatter(a_values_for_scatter, x_values_to_plot, 
                marker='.', 
                s=1,        
                linewidths=0,
                c='k') # 分岐図は黒で表示

# グラフの装飾
ax1.set_title('Bifurcation Diagram')
ax1.set_xlabel('Parameter a')
ax1.set_ylabel('Value of x')
ax1.set_xlim(a_s, a_f)
ax1.set_ylim(0, 1)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- 2. 時系列プロット (下の行 2, 3 に配置) ---
# 時系列プロットは、分岐図で計算済みのデータを使用 (最初の num_iterations_plot 回)

for idx, a_val in enumerate(a_values_to_examine):
    # a_listから a_val に最も近いインデックスを探す
    i_a = np.argmin(np.abs(a_list - a_val))
    
    # 計算済みのデータ x_list[i_a] の最初の num_iterations_plot 回を使用
    x_sequence = x_list[i_a][:num_iterations_plot]
    
    # サブプロットの定義
    # 2行3列のレイアウトの4番目, 5番目, 6番目にあたる位置
    ax = plt.subplot(2, 3, idx + 4) 
    
    # 時系列プロット
    ax.plot(range(num_iterations_plot), x_sequence, marker='o', linestyle='-', markersize=3, c='blue')
    
    # グラフの装飾
    ax.set_title(f'Time Series (a $\\approx$ {a_list[i_a]:.3f})') # 実際に使われたaの値を表示
    ax.set_xlabel('Iteration (t)')
    ax.set_ylabel('x(t)')
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--')

# グラフのレイアウト調整と表示
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.show()