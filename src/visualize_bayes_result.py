import matplotlib.pyplot as plt
import numpy as np


import csv
import argparse

parser = argparse.ArgumentParser(description='Parameters of virtual map.')
parser.add_argument('-i', dest='input_csv_filename', type=str, required=True ,help='input csv filename')




def main(infile):
    # 空のリストを4つ用意する
    x1 = []
    x2 = []
    y = []

    # CSVファイルを開く
    with open(infile, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # 各列の値をリストに追加する
            x1.append(float(row[0]))
            x2.append(float(row[1]))
            y.append(float(row[2]))


    visialization(x1, x2, y)

def visialization(x1, x2, y):
    # カラーマップの作成
    cmap = plt.cm.get_cmap('viridis')

    # 散布図のプロット
    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    ax.scatter(x1, x2, c=y)

    # カラーバーの追加
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(y)
    plt.colorbar(m)

    # グラフの設定
    ax.set_xlabel('lam1 and lam2')
    ax.set_ylabel('lam3')
    plt.title('Energy')

    plt.show()


if __name__ == "__main__":
    params = parser.parse_args()
    infile = params.input_csv_filename
    main(infile)
