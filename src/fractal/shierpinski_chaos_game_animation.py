import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# --- 1. 定数と初期設定 ---

# アニメーションで表示する点の総数
NUM_ITERATIONS = 5000 
# アニメーションの描画速度（フレームごとに追加する点数）
POINTS_PER_FRAME = 10 
# アニメーションのフレーム数
NUM_FRAMES = NUM_ITERATIONS // POINTS_PER_FRAME

# 三角形の3つの頂点
V = np.array([
    [0.0, 0.0],                       # 左下
    [1.0, 0.0],                       # 右下
    [0.5, np.sqrt(3) / 2]             # 上
])

# 描画データを格納する配列 (X, Y座標)
# 初期点を描画から除外するため、サイズは NUM_ITERATIONS
all_points = np.zeros((NUM_ITERATIONS, 2))


# --- 2. カオスゲームの点の座標を事前計算 ---

def generate_sierpinski_points():
    """カオスゲームの反復計算を行い、すべての点の座標を生成する"""
    P = np.array([0.0, 0.0])  # 初期点（どこでも良い）
    
    for i in range(NUM_ITERATIONS):
        # 3つの頂点からランダムに1つ選ぶ (0, 1, 2のいずれか)
        target_vertex_index = np.random.randint(3)
        target_V = V[target_vertex_index]
        
        # 現在の点 P と選んだ頂点 target_V の中点を計算し、新しい点とする
        P = P + (target_V - P) * 0.5
        
        # 座標を格納
        all_points[i] = P

# すべての点を計算
generate_sierpinski_points()


# --- 3. Matplotlibによるアニメーション設定 ---

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Sierpinski Triangle - Chaos Game Animation")
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

# 描画範囲の設定
# 頂点 V の最小/最大に合わせて少し余裕を持たせる
x_min, y_min = V.min(axis=0) - 0.1
x_max, y_max = V.max(axis=0) + 0.1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 描画オブジェクト（散布図）を初期化
# plot([], [], 'b.', markersize=2) は非推奨なので scatter を使用
scatter, = ax.plot([], [], 'o', markersize=2, color='blue', alpha=0.8)


def init():
    """アニメーション開始時の描画初期化関数"""
    scatter.set_data([], [])
    return scatter,

def animate(frame_number):
    """
    フレームごとの描画更新関数
    :param frame_number: 現在のフレーム番号
    """
    # 現在のフレームまでにプロットする点の数を計算
    end_index = (frame_number + 1) * POINTS_PER_FRAME
    
    # 描画する点のデータ
    x_data = all_points[:end_index, 0]
    y_data = all_points[:end_index, 1]
    
    # 描画オブジェクトを更新
    scatter.set_data(x_data, y_data)
    
    # フレーム番号を表示に追加
    ax.set_title(f"Sierpinski Triangle - Iteration: {end_index}")
    
    return scatter,

# --- 4. アニメーションの実行 ---

ani = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init,
    frames=NUM_FRAMES,      # 総フレーム数
    interval=50,            # 1フレームあたりのミリ秒数（50ms = 20fps）
    blit=True,              # 描画高速化（Trueが推奨）
    repeat=False            # 繰り返さない
)

plt.show()

# アニメーションをファイルに保存する場合は、以下の行をコメントアウト解除
# ani.save('sierpinski_chaos_game.gif', writer='imagemagick', fps=20)