# one dimensional collision problem (1次元衝突問題)
# Euler meathod

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

## 衝突解決の関数(めり込み解消ロジック付き)
def solve_collision(p1: Particle, p2: Particle, e: float)->bool:
    """
    質点p1とp2の一次元完全弾性衝突後の速度を計算し、更新する。
    p1: 質点1のオブジェクト
    p2: 質点2のオブジェクト
    e: 反発係数
    """
    m1, m2 = p1.m, p2.m
    v1i, v2i = p1.v, p2.v

    # 衝突後の速度の計算
    v1f = ( (m1 - e * m2) / (m1 + m2) ) * v1i + ( m2 * (1.0 + e)) / (m1 + m2) * v2i
    v2f = ( m1 * (1.0 + e)) / (m1 + m2) * v1i + ( (m2 - e * m1) / (m1 + m2) ) * v2i

    # 速度を更新
    p1.v = v1f
    p2.v = v2f
    
    print(f"  --> 衝突発生！v1:{v1i:.2f} -> {v1f:.2f}, v2:{v2i:.2f} -> {v2f:.2f}")

    # 合体状態になったかどうかをチェック (e=0、かつ相対速度がほぼゼロ)
    if e < 1e-9 and abs(p1.v - p2.v) < 1e-9:
        return True
    return False

def find_collision_time(p1: 'Particle', p2: 'Particle', dt: float, is_merged: bool) -> tuple[float, bool]:
    """
    次の時間刻み dt の間に衝突が発生するかどうかを判定し、
    衝突が発生する場合、現在の時刻から衝突時刻までの時間を返す。
    
    p1, p2: 質点オブジェクト
    dt: 時間刻み幅
    
    戻り値: (衝突までの時間 dt_col, 衝突が発生するかどうかのフラグ)
    """
    R = p1.r + p2.r
    x1, x2 = p1.x, p2.x
    v1, v2 = p1.v, p2.v

    distance = abs(x1 - x2)
    v_rel = v1 - v2
    
    # 1. 既に重なっている/接触している場合の処理
    if distance <= R:
        # **修正箇所**: 合体状態の場合、衝突イベントを報告しない
        if is_merged:
            return dt, False
            
        # 相対速度がほぼゼロの場合、衝突イベントを報告しない
        if abs(v_rel) < 1e-9:
            return dt, False 
        
        # 相対速度が残っている場合は、衝突イベントとして報告 (dt=0)
        return 0.0, True

    # 2. 互いに遠ざかっている場合は衝突しない
    # p1が右(x1>x2)でv1>v2 (遠ざかる) または p1が左(x1<x2)でv1<v2 (遠ざかる)
    if (x1 > x2 and v1 > v2) or (x1 < x2 and v1 < v2):
        return dt, False # dtを返すのは、このステップは通常通り進めてよいことを示す

    # 3. 衝突までの時間を逆算 (近づいている場合のみ)
    # 衝突するまでの相対距離 = 現在の距離 - 半径の和
    # 衝突までの時間 = 衝突するまでの相対距離 / 相対速度の大きさ
    # 相対速度がゼロの場合（平行移動）はゼロ除算を避ける
    if abs(v_rel) < 1e-9:
        return dt, False
        
    dt_to_col = (distance - R) / abs(v_rel)

    # 4. 衝突がこのステップ内 (0 < dt_to_col < dt) で起こるか判定
    if 0 < dt_to_col <= dt:
        return dt_to_col, True
    else:
        # 衝突がこのステップ内では起こらない (遠すぎる)
        return dt, False

## シミュレーション本体
def simulate(p1: Particle, p2: Particle, T_max: float=10.0, dt: float=0.01)->list[list[float]]:
    """
    シミュレーションを実行する
    T_max: シミュレーションの最大時間
    dt: 時間刻み幅
    e: 反発係数
    """
    t = 0.0
    e = 1.0
    is_merged = False # 粒子が合体状態になったかを示すフラグ

    print("--- シミュレーション開始 ---")
    
    # データを保存するためのリスト
    time_points = []
    x1_points = []
    x2_points = []
    
    step_count = 0 # ステップカウント数
    while t < T_max:
        
        # 衝突判定と衝突時刻までの時間を計算
        dt_to_col, collision_detected = find_collision_time(p1, p2, dt, is_merged)
        
        # 実際に時間を進める刻み幅
        dt_step = min(dt, T_max - t) # T_maxを超えないように調整
        
        # 1. 衝突が発生し、衝突時刻がこのステップ内にある場合
        if collision_detected and dt_to_col < dt_step:
            
            # (1) 衝突時刻まで進行
            p1.update_position(dt_to_col)
            p2.update_position(dt_to_col)
            t += dt_to_col
            
            # (2) 衝突解決 (速度更新とDepenetration)
            # **修正**: 衝突解決の戻り値で合体状態を判定
            is_merged = solve_collision(p1, p2, e)
            
            # (3) 残りの時間で進行
            dt_remain = dt_step - dt_to_col
            p1.update_position(dt_remain)
            p2.update_position(dt_remain)
            t += dt_remain

        # 2. 衝突がないか、衝突が次のステップ以降の場合
        else:
            p1.update_position(dt_step)
            p2.update_position(dt_step)
            t += dt_step

        step_count += 1
        # --- 3. データ記録（任意） ---
        if step_count % 100 == 0: # 100ステップごとに情報を表示
            time_points.append(t)
            x1_points.append(p1.x)
            x2_points.append(p2.x)
            # print(f"t={t:.2f}: P1.x={p1.x:.2f}, P2.x={p2.x:.2f}")

    print("--- シミュレーション終了 ---")
    return time_points, x1_points, x2_points


# --- 実行例 ---

# 質点1: 質量1.0, 位置0.0, 速度+1.0, 半径0.5
particle1 = Particle(mass=1.0, position=-50.0, velocity=2.0, radius=0.5) 
# 質点2: 質量1.0, 位置5.0, 速度-0.5, 半径0.5 (同じ質量なので速度が交換されるはず)
particle2 = Particle(mass=1.0, position=50.0, velocity=-1.0, radius=0.5)

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
ax.set_ylim(-30, 30)

# 縦横比を固定する: x軸とy軸の単位長を等しくする
# 'equal'に設定することで、x軸とy軸のスケールが1:1になる
ax.set_aspect('equal', adjustable='box')
ax.set_ylabel("") # y軸ラベルを空にする
ax.set_yticks([]) # y軸の目盛りを非表示にする
# 質点が動く線（y=0）を描画する
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
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
    blit=False,
    repeat=True,
)

# 凡例を再描画（blit=Trueだとinit_funcで描画された要素以外は消えるため）
ax.legend()

# アニメーションの表示
plt.show()