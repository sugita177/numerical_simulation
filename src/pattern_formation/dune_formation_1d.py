# dune formation
# one dimension
# 1次元の砂丘形成

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# --- シミュレーションパラメータ ---
NX = 200            # グリッドのセル数 (X軸方向)
ITERATIONS = 500    # 時間ステップ数
Q_IN = 0.5          # 供給される砂の量 (砂のフラックス)
P_THRESHOLD = 1.0   # 崩壊の閾値 (安息角に関連)

# --- 砂丘の初期化 ---
# 砂丘の高さ (1次元配列)
heights = np.zeros(NX)

# わずかな初期ノイズや障害物を設定
# 中央に小さな盛り上がりを作成
heights[int(NX/2 - 5) : int(NX/2 + 5)] = 0.1

# --- 可視化の設定 ---
fig, ax = plt.subplots()
line, = ax.plot(heights)
ax.set_ylim(-0.5, 3.0)
ax.set_title("Werner Model Dune Simulation")
ax.set_xlabel("Position (x)")
ax.set_ylabel("Height (h)")

# --- メインループ関数 ---
def update(frame):
    global heights

    # 1. 砂の跳躍・輸送ルール
    heights = apply_jumping_rule(heights, Q_IN)

    # 2. 砂の崩壊ルール (安息角による安定化)
    heights = apply_rolling_rule(heights, P_THRESHOLD)

    line.set_ydata(heights)
    ax.set_title(f"Time Step: {frame}")
    return line,

def apply_jumping_rule(h, Q_in):
    """
    風による砂の跳躍と輸送をシミュレーションします。
    """
    NX = len(h)
    
    # 輸送量 (フラックス) の計算:
    # 簡略化されたモデルでは、各セルから一定量の砂が風下に運ばれます。
    # 砂丘の勾配に応じた跳躍距離の変化（より高度なWernerモデルの機能）はここでは省略し、
    # 各セルからの輸送量を高さの変化として直接モデル化します。
    
    # h_new = h_current - (h_out - h_in)
    
    # 砂が削られる量 (h_out): 現在の高さに依存
    # Q_in が環境全体の砂の供給量を表す。ここでは簡略化し、すべてのセルが砂の源。
    
    # 砂の輸送フラックスを計算
    # 実際のWernerモデルでは勾配(h[i-1] - h[i])に応じて計算されますが、
    # ここでは最も簡単な形として、一定のフラックス Q_in を適用し、輸送量を表現します。
    
    # 輸送フラックス (Q)
    Q = np.full(NX, Q_in)
    
    # Q[i] はセル i-1 から i に移動する砂の量
    # 高さの変化 dH[i] = Q[i] - Q[i+1] (流入 - 流出)
    
    # 移動する砂の量を計算 (風下へ1セル分移動)
    # i のセルから削られる砂の量 (h[i] が Q_out の役割)
    sand_out = np.minimum(h, Q_in) # Q_in を超えて削ることはできない
    
    # 新しい高さ配列
    h_new = h.copy()
    
    # i から i+1 へ砂を輸送
    # h[i] から sand_out[i] を削る
    h_new -= sand_out
    
    # h[i+1] へ sand_out[i] を積もらせる
    # 周期境界条件 (右端から左端へ)
    h_new[1:] += sand_out[:-1]
    h_new[0] += sand_out[-1] # 周期境界条件 (最も右のセルから最も左のセルへ)
    
    return h_new

def apply_rolling_rule(h, P_threshold):
    """
    安息角の閾値を超えた砂の崩壊をシミュレーションします。
    """
    NX = len(h)
    h_new = h.copy()
    
    # 安息角の安定化が必要なくなるまでループ（一般には数回で十分）
    for _ in range(5): 
        # 隣接セルとの高さの差を計算 (i+1 と i の差)
        diff = h_new - np.roll(h_new, -1) # h[i] - h[i+1]
        
        # 崩壊が必要なセルを特定
        # 砂が i から i+1 へ崩れる条件: h[i] - h[i+1] > P_threshold
        # P_threshold は高さの差の最大許容値
        
        avalanche_mask = diff > P_threshold
        
        # 崩壊量 (崩れる砂の量)
        # 崩壊量 = (高さの差 - P_threshold) / 2
        # なぜ / 2 かというと、崩れた後の h[i] と h[i+1] の高さの差が P_threshold になるように調整するため
        amount_to_move = (diff[avalanche_mask] - P_threshold) / 2
        
        # 砂の移動
        # i から崩壊量を削る
        h_new[avalanche_mask] -= amount_to_move
        
        # i+1 へ崩壊量を積もらせる (周期境界条件を考慮)
        # np.roll(avalanche_mask, 1) は i+1 の位置のマスク
        roll_mask = np.roll(avalanche_mask, 1)
        h_new[roll_mask] += amount_to_move
        
    return h_new

# アニメーションの実行
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=ITERATIONS, 
    interval=50, 
    blit=True,
    repeat=False
)

plt.show()

# Jupyter Notebookや環境によっては以下のコマンドで保存も可能です。
# ani.save('dune_simulation.mp4', writer='ffmpeg')