import numpy as np
import matplotlib.pyplot as plt

def barnsley_fern(iterations=100000):
    """
    バーンスレイのシダをカオスゲームの原理で描画する。

    :param iterations: 繰り返しの回数（プロットする点の数）
    """
    # 1. 4つのアフィン変換の係数を定義
    # 各行が (a, b, c, d, e, f) に対応し、変換後の (x', y') は以下で計算される:
    # x' = a*x + b*y + e
    # y' = c*x + d*y + f
    
    # [確率, a, b, c, d, e, f]
    transforms = np.array([
        # 変換 1: 茎 (確率 1%)
        [0.01, 0.0,   0.0,  0.0, 0.16, 0.0,  0.0],
        # 変換 2: 右下の小葉 (確率 7%)
        [0.07, 0.2,  -0.26, 0.23, 0.22, 0.0,  1.6],
        # 変換 3: 左下の小葉 (確率 7%)
        [0.07, -0.15, 0.28, 0.26, 0.24, 0.0,  0.44],
        # 変換 4: 継続する大きな葉 (確率 85%)
        [0.85, 0.85,  0.04, -0.04, 0.85, 0.0, 1.6]
    ])
    
    # 累積確率を計算 (ランダム選択に使用)
    probabilities = transforms[:, 0]
    
    # 2. 初期設定
    P = np.array([0.0, 0.0])  # 初期点 (x, y)
    
    # 描画する点の座標を格納するための配列
    points = np.zeros((iterations, 2))
    
    # 3. 反復処理（カオスゲーム）
    for i in range(iterations):
        # 3-1. 確率に基づいて4つの変換から1つを選択
        # np.random.choice を使い、probabilitiesに指定された確率でインデックスを選ぶ
        idx = np.random.choice(4, p=probabilities)
        
        # 選択された変換の係数を取得
        a, b, c, d, e, f = transforms[idx, 1:]
        
        # 3-2. アフィン変換を適用して新しい点 P を計算
        x, y = P[0], P[1]
        
        # x' = a*x + b*y + e
        new_x = a * x + b * y + e
        
        # y' = c*x + d*y + f
        new_y = c * x + d * y + f
        
        # 新しい座標に更新
        P = np.array([new_x, new_y])
        
        # 3-3. 新しい点を記録
        points[i] = P

    # 4. 描画
    plt.figure(figsize=(6, 10)) # シダに合わせて縦長のサイズを設定
    # すべての点をプロット (マーカーサイズを非常に小さくする)
    plt.scatter(points[:, 0], points[:, 1], s=0.1, color='green') 
    
    # 軸を非表示
    plt.axis('off')
    plt.title(f'Barnsley Fern Fractal ({iterations} iterations)')
    plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を1:1に設定
    plt.show()

if __name__ == '__main__':
    # 10万回程度の繰り返しで、シダの形状が明確に現れます
    barnsley_fern(iterations=100000)