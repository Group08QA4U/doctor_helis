import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


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
    #fig = plt.figure()
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,8))

    #ax = fig.add_subplot(projection='3d')
    #ax1 = fig.add_subplot()
    ax1.scatter(x1, x2, c=y)

    # カラーバーの追加
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(y)
    plt.colorbar(m)

    # グラフの設定
    ax1.set_xlabel('X1(lam1 and lam2)')
    ax1.set_ylabel('X2(lam3)')
    ax1.set_xlim(35,100)
    ax1.set_ylim(0,20)
    plt.title('Energy')



    X1, X2 = np.meshgrid(np.unique(x1), np.unique(x2))
    Y = griddata((x1, x2), y, (X1, X2))
    


    cp = ax2.contourf(X1, X2,Y)
    ax2.set_xlabel('X1(lam1 and lam2)')
    ax2.set_ylabel('X2(lam3)')
    ax2.set_xlim(35,100)
    ax2.set_ylim(0,20)
    #plt.title('Contour plot of 3D data')
    #plt.colorbar(cp)


    plt.show()
    return 

if __name__ == "__main__":
    params = parser.parse_args()
    infile = params.input_csv_filename
    main(infile)
