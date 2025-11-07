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

# --- 可視化：分岐図の描画 ---
N_transient = 100  # 過渡期として無視する反復回数
N_plot = Nx - N_transient 

plt.figure(figsize=(10, 6))

for i_a in range(Na):
    a_val = a_list[i_a]
    
    # 過渡期後のxの値を取得
    x_values_to_plot = x_list[i_a][N_transient:]
    
    # 'a_val' の数だけ x の値のリストを作成
    a_values_for_scatter = np.full(N_plot, a_val)
    
    # 散布図としてプロット (点が小さく重なって線状に見える)
    plt.scatter(a_values_for_scatter, x_values_to_plot, 
                marker='.', 
                c='b',
                s=10,        
                linewidths=0) 

# グラフの装飾
plt.title('Bifurcation Diagram of the Logistic Map')
plt.xlabel('Parameter a')
plt.ylabel('Value of x')
plt.xlim(a_s, a_f)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
