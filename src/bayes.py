from skopt import gp_minimize
from skopt.space import Real, Space
import numpy as np
import optimizer
import csv

csv_data = []

# 最小化したい関数の定義
def objective(x):
    p, v = optimizer.bayes(x)
    print('type(p)',type(p))
    csv_data.append([p[0],p[1],v])
    return v

# 探索範囲の設定
space = Space([
    #Real(35.0, 100.0, name='x1'),
    #Real(0.0, 10.0, name='x2')
    Real(1.0, 50.0, name='x1'),
    Real(1.0, 50.0, name='x2')
])

#map_relocation = 5
#for i in range(map_relocation):
# ベイズ最適化の実行
result = gp_minimize(
    func=objective,  # 最小化したい関数
    dimensions=space,  # 探索範囲
    acq_func='EI',  # 獲得関数
    n_calls=100,  # 試行回数
    random_state=1111  # 再現性のためのランダムシード
)

# 結果の表示
print('the best lamdas:', result.x, 'energy:', result.fun)

with open('bayes_result.csv', 'w', ) as f:
    writer = csv.writer(f)
    writer.writerows(csv_data)


