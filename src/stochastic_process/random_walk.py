# random_walk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math

N: int = 10_000
x_list = np.zeros(N)
y_list = np.zeros(N)
x_list[0] = 0.0
y_list[0] = 0.0

for i in range(N-1):
    x: float = x_list[i]
    y: float = y_list[i]
    angle: float = random.random() * 2 * math.pi

    x += np.cos(angle)
    y += np.sin(angle)
    x_list[i+1] = x
    y_list[i+1] = y

# 以下はGeminiを利用して作成した
# 間引きレートを定義
skip_steps = 10 # 10ステップごとに描画
N_frames = N // skip_steps # 新しいフレーム総数

# アニメーションのための設定
x_length = max(abs(x_list.min()), abs(x_list.max()))
y_length = max(abs(y_list.min()), abs(y_list.max()))
fig, ax = plt.subplots(figsize=(8.0, 6.0)) # figとaxを分けて取得
ax.set_xlabel("x")
ax.set_xlim(-x_length, x_length)
ax.set_ylabel("y")
ax.set_ylim(-y_length, y_length)
ax.grid(True)


# 描画要素の初期化
# 粒子の軌跡全体を格納するLine2Dオブジェクト
line, = ax.plot([], [], color="b", label="Trajectory")
# 現在の粒子の位置をマークするScatterオブジェクト
point, = ax.plot([], [], 'o', color="r", markersize=8, label="Particle Position")

# アニメーションの初期化関数
def init():
    line.set_data([], [])
    point.set_data([], [])
    return line, point

# アニメーションの更新関数
# i はフレーム番号
def update(i):
    # 描画するデータのインデックスを計算
    data_index = i * skip_steps

    # 軌跡は data_index までを描画
    x_data = x_list[:data_index+1]
    y_data = y_list[:data_index+1]

    line.set_data(x_data, y_data)
    # 現在の位置も data_index の点を描画
    point.set_data([x_list[data_index]], [y_list[data_index]])

    # タイムスタンプをグラフに追加する場合は以下の行を使用
    # ax.set_title(f"Time: {t_list[data_index]:.2f} s")

    return line, point

# アニメーションの作成
# frames: 全フレーム数
# interval: フレーム間のミリ秒
# blit: True にすると、変更された要素のみを再描画するため高速化される
ani = FuncAnimation(
    fig, 
    update, 
    frames=N_frames, # 間引き後のフレーム数を使用
    init_func=init, 
    interval=100, # 指定した数値のms ごとに更新 (アニメーションの速度を調整)
    blit=True,
    repeat=False # 一番最後のフレームで止める
)

# 凡例を再描画（blit=Trueだとinit_funcで描画された要素以外は消えるため）
ax.legend()

# アニメーションの表示
plt.show()

# アニメーションをファイルに保存する場合は、以下を使用 (例: mp4)
# ani.save('coulomb_motion.mp4', writer='ffmpeg', fps=30)