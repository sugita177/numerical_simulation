# cellular automata
# traffic_flow
# rule 184

import numpy as np
import matplotlib.pyplot as plt

N: int = 50
R: int = 8
MAX_TIME: int = 50
RULE_NO: int = 184

def set_rule(rule, rule_no):
    for i in range(R):
        rule[i] = rule_no % 2
        rule_no = rule_no // 2

def calc_next_state(ca, rule):
    next_ca = np.zeros(N, dtype=int)
    for i in range(0, N):
        # periodic boundary condition
        prev_index = (i-1) % N
        next_index = (i+1) % N
        next_ca[i] = rule[4 * ca[next_index] + 2 * ca[i] + ca[prev_index]]
    return next_ca

ca = np.random.randint(0, 2, size=N)
rule = np.zeros(R, dtype=int)
set_rule(rule, RULE_NO)

# 履歴を保存する配列
# MAX_TIME + 1 の行 (初期状態を含む) と N の列
history = np.zeros((MAX_TIME + 1, N), dtype=int)

# 最初の状態を保存
history[0, :] = ca

for i in range(MAX_TIME):
    ca = calc_next_state(ca, rule)
    history[i + 1, :] = ca # 次の状態を保存

# 可視化コードの続き
plt.figure(figsize=(10, 8))

# imshowで2次元配列を画像として表示
# interpolation='none'は、セルがカクカクしたピクセルとして表示されるようにします。
# cmap='binary'は、0を白、1を黒で表示します（一般的なCAの慣習）。
plt.imshow(history, interpolation='none', cmap='binary')

plt.title(f'Cellular Automata Rule {RULE_NO} (N={N}, T={MAX_TIME})')
plt.xlabel('Cell Index')
plt.ylabel('Time Step')

# 軸をわかりやすくするために設定
plt.xticks(np.arange(0, N, 10))
plt.yticks(np.arange(0, MAX_TIME + 1, 10))

plt.colorbar(label='State (1=Active/Car, 0=Inactive/Empty)')
plt.show()