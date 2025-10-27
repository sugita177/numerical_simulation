import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. 定数と初期設定 ---

# アニメーションで表示する点の総数
NUM_ITERATIONS = 50000 
# アニメーションの描画速度（フレームごとに追加する点数）
POINTS_PER_FRAME = 50 
# アニメーションのフレーム数
NUM_FRAMES = NUM_ITERATIONS // POINTS_PER_FRAME

# 4つのアフィン変換の係数を定義
# 各行が (確率, a, b, c, d, e, f) に対応
transforms_data = np.array([
    # 変換 1: 茎 (確率 1%)
    [0.01, 0.0,   0.0,  0.0, 0.16, 0.0,  0.0],
    # 変換 2: 右下の小葉 (確率 7%)
    [0.07, 0.2,  -0.26, 0.23, 0.22, 0.0,  1.6],
    # 変換 3: 左下の小葉 (確率 7%)
    [0.07, -0.15, 0.28, 0.26, 0.24, 0.0,  0.44],
    # 変換 4: 継続する大きな葉 (確率 85%)
    [0.85, 0.85,  0.04, -0.04, 0.85, 0.0, 1.6]
])

# 確率
probabilities = transforms_data[:, 0]

# アフィン変換行列とそのオフセットベクトルを抽出
# transforms_matrices は 4x2x2 の配列 (2x2行列が4つ)
transforms_matrices = np.array([
    [[t[1], t[2]], [t[3], t[4]]] for t in transforms_data
])
# offsets は 4x2 の配列 (2要素ベクトルが4つ)
offsets = np.array([
    [t[5], t[6]] for t in transforms_data
])

# 描画データを格納する配列 (X, Y座標)
all_points = np.zeros((NUM_ITERATIONS, 2))


# --- 2. バーンスレイのシダの座標を事前計算 ---

def generate_barnsley_fern_points():
    """バーンスレイのシダの反復計算を行い、すべての点の座標を生成する"""
    P = np.array([0.0, 0.0])  # 初期点 (x, y)
    
    for i in range(NUM_ITERATIONS):
        # 確率に基づいて4つの変換から1つを選択
        idx = np.random.choice(4, p=probabilities)
        
        # 選択された変換の行列とオフセットを取得
        matrix = transforms_matrices[idx]
        offset = offsets[idx]
        
        # アフィン変換を適用して新しい点 P を計算
        # P_new = M @ P + offset (NumPyの行列積)
        P = np.dot(matrix, P) + offset
        
        # 座標を格納
        all_points[i] = P

# すべての点を計算
generate_barnsley_fern_points()


# --- 3. Matplotlibによるアニメーション設定 ---

fig, ax = plt.subplots(figsize=(6, 10)) # シダに合わせて縦長のサイズ
ax.set_title("Barnsley Fern - Chaos Game Animation")
ax.set_aspect('equal', adjustable='box')
ax.axis('off')

# 描画範囲の設定
# シダの典型的な範囲 (-3 to 3 for x, -1 to 11 for y)
ax.set_xlim(-3, 3)
ax.set_ylim(-1, 11)

# 描画オブジェクト（散布図）を初期化
scatter, = ax.plot([], [], 'o', markersize=0.5, color='green', alpha=0.8) # マーカーサイズを小さく


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
    ax.set_title(f"Barnsley Fern - Iteration: {end_index}")
    
    return scatter,

# --- 4. アニメーションの実行 ---

ani = animation.FuncAnimation(
    fig, 
    animate, 
    init_func=init,
    frames=NUM_FRAMES,      # 総フレーム数
    interval=1,            # 1フレームあたりのミリ秒数（1ms = 1000fps、超高速！）
    blit=False,              # 描画高速化
    repeat=False            # 繰り返さない
)

plt.show()

# アニメーションをファイルに保存する場合は、以下の行をコメントアウト解除
# ani.save('barnsley_fern_chaos_game.gif', writer='imagemagick', fps=60)