import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from scipy.spatial import cKDTree # 大規模なシミュレーションではKDTreeを使うと高速化できる

# --- シミュレーション定数 ---
MAX_X = 400.0  # フィールドの最大幅
MAX_Y = 400.0  # フィールドの最大高さ
DEL_T = 0.1    # 変更: 時間刻みを 0.05 から 0.1 に増加 (移動量が増える)
TIME_STEP = 500 # シミュレーションステップ数
AGENT_NUMBER = 100 # エージェントの数

# --- Boidルールのパラメータ ---
R_VIEW = 20.0  # 視野半径
R_SEPARATION = 5.0 # 分離の半径

# 各ルールの重み（加速度への影響度）
# 変更: 全体的に重みを増加させ、操舵力を強くした
W_SEPARATION = 3.0 # 分離の重み
W_ALIGNMENT = 1.5  # 整列の重み
W_COHESION = 1.0   # 結合の重み

MAX_SPEED = 10.0   # 変更: 最大速度を 5.0 から 10.0 に増加 (移動距離が増える)
MIN_SPEED = 1.0    # 変更: 最小速度を 0.5 から 1.0 に増加 (最低限の移動を保証)

# --- Boidの状態を保持するグローバル変数 ---
positions = np.zeros((AGENT_NUMBER, 2))  # (x, y) 座標
velocities = np.zeros((AGENT_NUMBER, 2)) # (vx, vy) 速度

# --------------------------------------------------
# 初期化
# --------------------------------------------------
def initialize_boids():
    """ボイドの位置と速度を初期化する"""
    global positions, velocities
    positions[:, 0] = np.random.rand(AGENT_NUMBER) * MAX_X
    positions[:, 1] = np.random.rand(AGENT_NUMBER) * MAX_Y
    
    # 初期速度もランダムだが、MAX_SPEEDを超えないように制限
    velocities = (np.random.rand(AGENT_NUMBER, 2) - 0.5) * 2.0 * MAX_SPEED
    
    # 全ての初期速度に最小速度を適用
    for i in range(AGENT_NUMBER):
        velocities[i] = clip_velocity(velocities[i])

# --------------------------------------------------
# 近傍計算 (変更なし)
# --------------------------------------------------
def get_distances_and_neighbors(i):
    """
    ボイドiに対する全てのボイドとの距離を計算し、近傍ボイドのマスクを返す。
    周期境界条件を考慮した距離計算（トーラス空間）を行う。
    """
    rel_pos = positions - positions[i]
    
    # 周期境界条件の適用
    rel_pos[:, 0] = np.where(rel_pos[:, 0] > MAX_X / 2, rel_pos[:, 0] - MAX_X, rel_pos[:, 0])
    rel_pos[:, 0] = np.where(rel_pos[:, 0] < -MAX_X / 2, rel_pos[:, 0] + MAX_X, rel_pos[:, 0])
    rel_pos[:, 1] = np.where(rel_pos[:, 1] > MAX_Y / 2, rel_pos[:, 1] - MAX_Y, rel_pos[:, 1])
    rel_pos[:, 1] = np.where(rel_pos[:, 1] < -MAX_Y / 2, rel_pos[:, 1] + MAX_Y, rel_pos[:, 1])

    distances = np.linalg.norm(rel_pos, axis=1) # ユークリッド距離

    # 視野半径R_VIEW内のボイド (自身を除く)
    near_by_mask = (distances < R_VIEW) & (distances > 1e-6) 

    return distances, near_by_mask, rel_pos

# --------------------------------------------------
# Boidルールの適用 (係数のみ変更)
# --------------------------------------------------
def apply_boid_rules(i, distances, near_by_mask, rel_pos):
    """
    ボイドiに対して、分離、整列、結合の3つのルールに基づく加速度を計算する
    """
    
    near_by_pos = positions[near_by_mask]
    near_by_vel = velocities[near_by_mask]
    
    if len(near_by_pos) == 0:
        return np.zeros(2)

    # --- 1. 分離 (Separation) ---
    separation_force = np.zeros(2)
    separation_mask = (distances < R_SEPARATION) & near_by_mask
    separation_neighbors = rel_pos[separation_mask]
    
    if separation_neighbors.size > 0:
        # 近いほど強く反発させるため、逆方向のベクトルを合計
        separation_force = -separation_neighbors.sum(axis=0)
    
    # --- 2. 整列 (Alignment) ---
    avg_neighbor_vel = near_by_vel.mean(axis=0)
    alignment_force = avg_neighbor_vel - velocities[i]

    # --- 3. 結合 (Cohesion) ---
    avg_neighbor_pos = near_by_pos.mean(axis=0)
    cohesion_force = avg_neighbor_pos - positions[i]

    # --- 最終的な加速度の合計 ---
    # 変更: W_SEPARATION, W_ALIGNMENT, W_COHESION の値を大きくしている
    acceleration = (
        W_SEPARATION * separation_force +
        W_ALIGNMENT * alignment_force +
        W_COHESION * cohesion_force
    )
    
    return acceleration

# --------------------------------------------------
# 速度の制限 (変更なし)
# --------------------------------------------------
def clip_velocity(vel):
    """速度ベクトルが最大速度と最小速度の間に収まるように調整する"""
    speed = np.linalg.norm(vel)
    
    if speed > MAX_SPEED:
        return vel / speed * MAX_SPEED
    elif speed < MIN_SPEED and speed > 1e-6:
        return vel / speed * MIN_SPEED
    elif speed < 1e-6:
        random_direction = np.random.rand(2) - 0.5
        random_direction /= np.linalg.norm(random_direction)
        return random_direction * MIN_SPEED
    
    return vel

# --------------------------------------------------
# シミュレーション更新 (変更なし)
# --------------------------------------------------
def update_simulation():
    """シミュレーションを1ステップ進める"""
    global positions, velocities
    
    current_velocities = velocities.copy()
    
    # 1. 加速度の計算と速度の更新
    for i in range(AGENT_NUMBER):
        distances, near_by_mask, rel_pos = get_distances_and_neighbors(i)
        acceleration = apply_boid_rules(i, distances, near_by_mask, rel_pos)
        
        new_vel = current_velocities[i] + acceleration * DEL_T
        
        # 速度のクリップ
        velocities[i] = clip_velocity(new_vel)
        
    # 2. 位置の更新
    positions += velocities * DEL_T
    
    # 3. 周期境界条件の適用
    positions[:, 0] = (positions[:, 0] + MAX_X) % MAX_X
    positions[:, 1] = (positions[:, 1] + MAX_Y) % MAX_Y
    
    return positions

# --------------------------------------------------
# メイン処理 (変更なし)
# --------------------------------------------------
def main():
    initialize_boids()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, MAX_X)
    ax.set_ylim(0, MAX_Y)
    ax.set_aspect('equal', adjustable='box')

    # 描画用の単位ベクトルを計算
    # 速度の大きさを計算
    speeds = np.linalg.norm(velocities, axis=1)
    # ゼロ割を避けるため、速度ゼロのボイドの speed を 1 に設定 (単位ベクトルを計算するため)
    speeds[speeds == 0] = 1.0 
    # 単位ベクトル (方向成分のみ)
    unit_vectors = velocities / speeds[:, np.newaxis]
    
    # 速度ベクトルを表示するために quivers (矢印) を使用
    # scale の値も速度に合わせて調整すると良い
    q = ax.quiver(positions[:, 0], positions[:, 1],
                  unit_vectors[:, 0], unit_vectors[:, 1], # U, V に単位ベクトルを使用
                  scale=0.1, units='xy', color='green', alpha=0.8, 
                  headwidth=5, headlength=7) 
    
    def animate(frame):
        # 1ステップのシミュレーション更新
        update_simulation()

        # 描画の際にも単位ベクトルを計算し直す
        speeds = np.linalg.norm(velocities, axis=1)
        speeds[speeds == 0] = 1.0 
        unit_vectors = velocities / speeds[:, np.newaxis]
        
        # quivers (矢印) のデータを更新
        q.set_offsets(positions)
        q.set_UVC(unit_vectors[:, 0], unit_vectors[:, 1]) # 単位ベクトルをセット
        
        ax.set_title(f"Time Step: {frame}/{TIME_STEP}")
        return q,
    
    # アニメーションの作成と実行
    # interval も DEL_T * 1000 に合わせているため、DEL_Tが増加した分、描画間隔も広がる
    ani = animation.FuncAnimation(
        fig, animate, frames=TIME_STEP+1, interval=DEL_T * 1000, 
        blit=False, repeat=False
    )
    
    plt.show()

if __name__ == "__main__":
    main()