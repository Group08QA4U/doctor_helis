from skopt import gp_minimize
from skopt.space import Real, Space
import numpy as np
import optimizer

# 最小化したい関数の定義
def objective(x):
    return optimizer.bayes(x)

# 探索範囲の設定
space = Space([
    Real(30.0, 50.0, name='x1'),
    Real(30.0, 50.0, name='x2'),
    Real(1.0, 5.0, name='x3')
])

map_relocation = 5
for i in range(map_relocation):
    # ベイズ最適化の実行
    result = gp_minimize(
        func=objective,  # 最小化したい関数
        dimensions=space,  # 探索範囲
        acq_func='EI',  # 獲得関数
        n_calls=20,  # 試行回数
        random_state=1111  # 再現性のためのランダムシード
    )

    # 結果の表示
    print('map#{:02}'.format(i),'the best lamdas:', result.x, 'energy:', result.fun)

