import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


import csv
import argparse

parser = argparse.ArgumentParser(description='Parameters of virtual map.')
parser.add_argument('-q1', dest='input_csv_q1', type=str, required=True ,help='input csv Q1')
parser.add_argument('-q2', dest='input_csv_q2', type=str, required=True ,help='input csv Q2')
parser.add_argument('-q3', dest='input_csv_q3', type=str, required=True ,help='input csv Q3')




def main(q1,q2,q3):
    qubos = [q1,q2,q3]
    titles = ['Q1','Q2','Q3']
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,8))
    axes = [ax1, ax2, ax3]

    for qubo, ax, title in zip(qubos,axes,titles):
        # 空のリストを4つ用意する
        x1 = []
        x2 = []
        y = []

        # CSVファイルを開く
        with open(qubo, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                # 各列の値をリストに追加する
                x1.append(float(row[0]))
                x2.append(float(row[1]))
                y.append(float(row[2]))
        print(title,np.mean(y))

        visialization(ax, x1, x2, y, title)

    # カラーマップの作成
    cmap = plt.cm.get_cmap('viridis')
    # カラーバーの追加
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(y)
    plt.colorbar(m)
    #plt.title('QUBO matrices')
    plt.show()

def visialization(ax, x1, x2, y, title):
    ax.set_title(title)

    # 散布図のプロット
    #fig = plt.figure()
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))

    #ax = fig.add_subplot(projection='3d')
    #ax1 = fig.add_subplot()
    ax.scatter(x1, x2, c=y)


    # グラフの設定
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    #ax1.set_ylim(0,20)
    return 

if __name__ == "__main__":
    params = parser.parse_args()
    q1 = params.input_csv_q1
    q2 = params.input_csv_q2
    q3 = params.input_csv_q3
    main(q1,q2,q3)
