import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Callable, Any

# ===============================================
# 既存のセルオートマタ（CA）の定義とロジック
# ===============================================

N: int = 50 
R: int = 8 
MAX_TIME: int = 5000
RULE_NO: int = 184

def set_rule(rule: np.ndarray, rule_no: int) -> None:
    """ルール番号からルール配列を設定する関数"""
    for i in range(R):
        rule[i] = rule_no % 2
        rule_no = rule_no // 2

def calc_next_state(ca: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """次の状態を計算する関数 (Rule 184)"""
    next_ca = np.zeros(N, dtype=int)
    for i in range(0, N):
        # 周期境界条件
        prev_index = (i - 1) % N
        next_index = (i + 1) % N
        
        # 3つのセル (左, 中央, 右) の状態を結合してインデックスを計算
        # インデックス: 4 * ca[左] + 2 * ca[中央] + ca[右]
        index = 4 * ca[prev_index] + 2 * ca[i] + ca[next_index]
        
        next_ca[i] = rule[index]
    return next_ca

# 初期設定
ca_initial = np.random.randint(0, 2, size=N)
rule = np.zeros(R, dtype=int)
set_rule(rule, RULE_NO)

# 履歴を生成 (アニメーション用)
history: List[np.ndarray] = [ca_initial]
current_ca = ca_initial.copy()

for _ in range(MAX_TIME):
    current_ca = calc_next_state(current_ca, rule)
    history.append(current_ca.copy())


# ===============================================
# アニメーションの作成 (三角形マーカーに変更)
# ===============================================

# N個のセルのインデックス（x軸の位置）
x_positions = np.arange(N)

fig, ax = plt.subplots(figsize=(12, 3))

# 初期プロット：すべてのセルを状態0（空）として表示
# グラフ全体に道路（レール）のような線を表示するため、y=0の水平線を使用
ax.hlines(0, -0.5, N - 0.5, colors='gray', linestyles='-', linewidth=2) 

# 車（状態1）を表す散布図。最初は空の状態で作成。
# y=0の位置に表示。s=300は丸のサイズ、c='blue'は色。
# ここを marker='>' に変更し、右向きの三角形にします。
scatter = ax.scatter([], [], s=300, c='blue', marker='>', zorder=5) 

# グラフの見た目の設定
ax.set_xlim(-0.5, N - 0.5) 
ax.set_ylim(-1, 1) 
ax.set_title(f'Cellular Automata Rule {RULE_NO} Traffic Flow (Time Step: 0)')
ax.set_xlabel('Cell Index (Road Position)')
ax.set_yticks([]) 
ax.set_xticks(np.arange(0, N, 5))
ax.grid(axis='x', linestyle=':', alpha=0.5)


def update(frame: int) -> List[Any]:
    """
    アニメーションの各フレームで呼び出される更新関数。
    frame: 現在のステップインデックス
    """
    state = history[frame]
    
    # 車（状態1）のセルインデックスを抽出
    car_indices = x_positions[state == 1]
    
    # 散布図のデータを更新
    # x軸: 車のある位置 (car_indices)
    # y軸: すべて0
    scatter.set_offsets(np.c_[car_indices, np.zeros_like(car_indices)])
    
    # タイトルを更新して現在のステップを表示
    ax.set_title(f'Cellular Automata Rule {RULE_NO} Traffic Flow (Time Step: {frame})')
    
    return [scatter, ax.title]


# アニメーションの作成
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=len(history), 
    interval=150,        
    blit=False,           
    repeat=False         
)

plt.show()