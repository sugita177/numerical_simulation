# one dimensional collision problem (1次元衝突問題)
# Runge Kutta meathod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

## 質点クラスの定義
class Particle:
    """質点を表すクラス（一次元）"""
    def __init__(self, mass, position, velocity, radius):
        self.m = mass          # 質量
        self.x = position      # 位置
        self.v = velocity      # 速度
        self.r = radius        # 半径 (衝突判定用)

    def update_position(self, dt):
        """位置を更新する"""
        self.x += self.v * dt

## 衝突解決の関数
def solve_collision(p1: Particle, p2: Particle):
    """
    質点p1とp2の一次元完全弾性衝突後の速度を計算し、更新する。
    p1: 質点1のオブジェクト
    p2: 質点2のオブジェクト
    """
    m1, m2 = p1.m, p2.m
    v1i, v2i = p1.v, p2.v

    # 衝突後の速度の計算
    v1f = ( (m1 - m2) / (m1 + m2) ) * v1i + ( (2 * m2) / (m1 + m2) ) * v2i
    v2f = ( (2 * m1) / (m1 + m2) ) * v1i + ( (m2 - m1) / (m1 + m2) ) * v2i

    # 速度を更新
    p1.v = v1f
    p2.v = v2f
    
    print(f"  --> 衝突発生！v1:{v1i:.2f} -> {v1f:.2f}, v2:{v2i:.2f} -> {v2f:.2f}")

## シミュレーション本体
def simulate(p1: Particle, p2: Particle, T_max: float=10.0, dt: float=0.01)->list[list[float]]:
    """
    シミュレーションを実行する
    T_max: シミュレーションの最大時間
    dt: 時間刻み幅
    """
    t = 0.0
    collision_radius = p1.r + p2.r # 衝突判定に必要な2質点の半径の和
    
    # 衝突が発生した後のフラグ。多重衝突を防ぐために使用
    collided_in_step = False

    print("--- シミュレーション開始 ---")
    
    # データを保存するためのリスト
    time_points = []
    x1_points = []
    x2_points = []
    
    while t < T_max:
        # --- 1. 衝突判定 ---
        distance = abs(p1.x - p2.x)

        # 質点間の距離が半径の和以下になったら衝突と判定
        if distance <= collision_radius and not collided_in_step:
            print(f"t={t:.2f}で衝突検出。")
            solve_collision(p1, p2)
            collided_in_step = True # このステップで既に衝突を解決した
            
            # 位置を少し戻して、正確な衝突瞬間に合わせることもできるが、
            # シンプルなシミュレーションでは速度更新のみで対応

        elif distance > collision_radius:
            # 衝突範囲を離れたら、次の衝突を許可
            collided_in_step = False 

        # --- 2. 時間発展（位置の更新） ---
        p1.update_position(dt)
        p2.update_position(dt)
        t += dt
        
        # --- 3. データ記録（任意） ---
        if int(t / dt) % 100 == 0: # 100ステップごとに情報を表示
            time_points.append(t)
            x1_points.append(p1.x)
            x2_points.append(p2.x)
            # print(f"t={t:.2f}: P1.x={p1.x:.2f}, P2.x={p2.x:.2f}")

    print("--- シミュレーション終了 ---")
    return time_points, x1_points, x2_points


# --- 実行例 ---

# 質点1: 質量1.0, 位置0.0, 速度+1.0, 半径0.5
particle1 = Particle(mass=1.0, position=-20.0, velocity=1.0, radius=0.5) 
# 質点2: 質量1.0, 位置5.0, 速度-0.5, 半径0.5 (同じ質量なので速度が交換されるはず)
particle2 = Particle(mass=1.0, position=20.0, velocity=-0.5, radius=0.5)

# 2. 質量が異なる場合
# particle1 = Particle(mass=1.0, position=-20.0, velocity=1.0, radius=0.5) 
# particle2 = Particle(mass=8.0, position=20.0, velocity=-0.5, radius=0.5) # 重い質点にぶつかる

# シミュレーション実行
t, x1, x2 = simulate(particle1, particle2, T_max=100.0, dt=0.01)

# -- アニメーション --
# グラフの準備
fig, ax = plt.subplots(figsize=(8.0, 6.0)) # figとaxを分けて取得
ax.set_xlabel("x")
x_abs_max = max(abs(min(min(x1), min(x2))), abs(max(max(x1), max(x2))))
ax.set_xlim(-x_abs_max, x_abs_max)
ax.set_ylabel("y")
ax.set_ylim(-30, 30)
ax.grid(True)
ax.set_title("Two Body Collision Problem in One Dimension")

# 現在の粒子の位置をマークするScatterオブジェクト
point1, = ax.plot([], [], 'o', color="b", markersize=8, label="Particle1 Position")
point2, = ax.plot([], [], 'o', color="r", markersize=8, label="Particle2 Position")


# アニメーションの初期化関数
def init():
    point1.set_data([], [])
    point2.set_data([], [])
    return point1, point2

# アニメーションの更新関数
# i はフレーム番号
def update(i):

    # 現在の位置も data_index の点を描画
    point1.set_data([x1[i]], [0])
    point2.set_data([x2[i]], [0])


    # タイムスタンプをグラフに追加する場合は以下の行を使用
    # ax.set_title(f"Time: {t_list[data_index]:.2f} s")

    return point1, point2

# アニメーションの作成
# frames: 全フレーム数
# interval: フレーム間のミリ秒
# blit: True にすると、変更された要素のみを再描画するため高速化される
ani = FuncAnimation(
    fig, 
    update, 
    frames=len(t), # 間引き後のフレーム数を使用
    init_func=init, 
    interval=50, # 指定した数値のms ごとに更新 (アニメーションの速度を調整)
    blit=True,
    repeat=True,
)

# 凡例を再描画（blit=Trueだとinit_funcで描画された要素以外は消えるため）
ax.legend()

# アニメーションの表示
plt.show()