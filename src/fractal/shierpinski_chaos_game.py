import numpy as np
import matplotlib.pyplot as plt

def chaos_game_sierpinski(iterations=50000):
    """
    カオスゲームのアルゴリズムを用いてシェルピンスキーの三角形を描画する。

    :param iterations: 繰り返しの回数（プロットする点の数）
    """
    # 1. 初期設定: 3つの固定点（三角形の頂点）
    # 正三角形になるように座標を設定
    V = np.array([
        [0.0, 0.0],       # 頂点1 (左下)
        [1.0, 0.0],       # 頂点2 (右下)
        [0.5, np.sqrt(3)/2] # 頂点3 (上)
    ])
    
    # 2. 初期点: 3つの頂点のいずれか、または内部の任意の点
    # ここでは、頂点の一つを初期点として使用
    P = np.array([0.0, 0.0])
    
    # 描画する点の座標を格納するための配列
    points = np.zeros((iterations, 2))
    
    # 3. 反復処理（カオスゲーム）
    for i in range(iterations):
        # 3-1. 3つの頂点からランダムに一つ選ぶ (0, 1, 2のいずれか)
        # 乱数を使って、どの頂点に近づくかを決定する
        target_vertex_index = np.random.randint(3)
        target_V = V[target_vertex_index]
        
        # 3-2. 現在の点Pと選んだ頂点target_Vの中点を計算し、新しい点Pとする
        # 新しい点 = P + (target_V - P) * 0.5
        P = P + (target_V - P) * 0.5
        
        # 3-3. 新しい点を記録
        points[i] = P

    # 4. 描画
    plt.figure(figsize=(8, 8))
    # すべての点をプロット (マーカーサイズを小さくする)
    plt.scatter(points[:, 0], points[:, 1], s=0.1, color='blue') 
    
    # 軸の目盛りを非表示にして、三角形を際立たせる
    plt.axis('off')
    plt.title(f'Sierpinski Triangle via Chaos Game ({iterations} iterations)')
    plt.gca().set_aspect('equal', adjustable='box') # アスペクト比を1:1に設定
    plt.show()

if __name__ == '__main__':
    # 5万回程度の繰り返しで美しいフラクタルが観察できます
    chaos_game_sierpinski(iterations=50000)