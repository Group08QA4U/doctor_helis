from skopt import gp_minimize
from skopt.space import Real, Space
import numpy as np
import optimizer
import csv

csv_data = []


import argparse

parser = argparse.ArgumentParser(description='Parameters of virtual map.')
parser.add_argument('-q', dest='qa_retry_cnt', type=int, default=3, help='QA retry count')
parser.add_argument('-width', dest='width', type=int, default=20000, help='width')
parser.add_argument('-height', dest='height', type=int, default=20000, help='height')
parser.add_argument('-p', dest='patients', type=int, default=8, help='num of patients')
parser.add_argument('-a', dest='ambulance', type=int, default=9, help='num of ambulances')
parser.add_argument('-r', dest='rendezvous_points', type=int, default=11, help='num of rendezvous points')
parser.add_argument('-d', dest='doctorhelis', type=int, default=8, help='num of doctor helis')



def main(qa_trial_count, width, height, num_of_patients, num_of_fire_departments, num_of_rendezvous_points, num_of_basehospitals):
    # 最小化したい関数の定義
    def objective(x):
        #p, v = optimizer.bayes(x)
        p, v = optimizer.bayes(x, qa_trial_count, width, height, num_of_patients, num_of_fire_departments, num_of_rendezvous_points, num_of_basehospitals, use_d_wave=False)

        csv_data.append([p[0],p[1],v])
        return v

    params = [qa_trial_count,width,height,num_of_patients,num_of_fire_departments,\
              num_of_rendezvous_points,num_of_basehospitals]


    # 探索範囲の設定
    space = Space([
        Real(35.0, 100.0, name='x1'),
        Real(0.0, 20.0, name='x2')
    ])

    result = gp_minimize(
        func=objective,  # 最小化したい関数
        dimensions=space,  # 探索範囲
        acq_func='EI',  # 獲得関数
        n_calls=10,  # 試行回数
        random_state=1111  # 再現性のためのランダムシード
    )

    # 結果の表示
    print('the best lamdas:', result.x, 'energy:', result.fun)


    csv_filename = 'bayes_result' + '.' + 'q' + str(params[0]) + \
              '.' + 'w' + str(params[1]) + '.' + 'h' + str(params[2]) + \
              '.' + 'p' + str(params[3]) + '.' + 'a' + str(params[4]) + \
              '.' + 'r' + str(params[5]) + '.' + 'd' + str(params[6]) + '.csv'

    print('Save as',csv_filename)
    with open(csv_filename, 'w', ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print('Done')    

if __name__ == "__main__":
    params = parser.parse_args()
    q = params.qa_retry_cnt
    w = params.width
    h = params.height
    p = params.patients
    a = params.ambulance
    r = params.rendezvous_points
    d = params.doctorhelis
    main(q, w, h, p, a, r, d)
