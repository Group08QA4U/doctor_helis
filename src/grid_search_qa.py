# -*- coding: utf-8 -*-
import os
import datetime
import argparse
import copy
import csv

import time
from multiprocessing import Pool

from tqdm import tqdm

import optimizer
import animation
import parameters


parser = argparse.ArgumentParser(description='Hyper parameter.')
parser.add_argument('-cpu', dest='cpu', type=int, default=15, help='num of cpu core')
parser.add_argument('-csv_file', dest='csv_file', type=str, help='target data', required=True)
parser.add_argument('-save_dir', dest='save_dir', default='output', type=str, help='Directory for output files')


def write_result_csv(out_filename, results_data):
    labels = list(results_data[0].keys())

    try:
        with open(out_filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=labels)
            writer.writeheader()
            for elem in results_data:
                writer.writerow(elem)
    except IOError:
        print("I/O error")



def wraper(parameters_list):
    life_saving_resources_params, hyper_params = parameters_list
    optimizer.grid_search(life_saving_resources_params, hyper_params)


    params_str, params_dict = life_saving_resources_params.get_title_from_params()
    print(str('{:.4f}'.format(acc)), params_str)
    return acc, params_str, params_dict

def main(life_saving_resources_params):
    p = Pool(n_cpu)

    parameters_list = []
    print('Preparing a gird search. Please wait... ')
    with tqdm() as pbar:
        while life_saving_resources_params.set_next_params():
            parameters_list.append(copy.deepcopy((life_saving_resources_params,hyper_params)))
            pbar.update(1)

    result_data = []
    results = p.map(wraper, parameters_list)
    print('Results of grid search for', APP_NAME)
    for result in sorted(results, reverse=True):
        acc, params_str, params_dict = result
        result_dict = {}
        result_dict.update({'acc':acc})
        result_dict.update(params_dict)
        result_data.append(result_dict)
        print(acc, params_str)


    result_csv_filename = os.path.join(save_dir, 'results_grid_search_' + APP_NAME + '.csv')
    write_result_csv(result_csv_filename, result_data)
    print('Save result csv to ',result_csv_filename)


if __name__ == '__main__':
    params = parser.parse_args()

    save_dir = os.path.join(params.save_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)
    n_cpu = params.cpu
    csv_file = params.csv_file

    life_saving_resources_params = parameters.LifeSavingResourcesParams()
    hyper_params = parameters.HyperParams()

    try:
        main(life_saving_resources_params)
    except KeyboardInterrupt:
        print('closeing...')
    exit()






def main():
    num_of_fire_departments = 14 #救急車
    num_of_basehospitals = 14 #ドクターヘリ
    width = 86000
    height = 86000

    for num_of_patients in [4,7,14]:
      for num_of_rendezvous_points in [20,40,80,160,320,640,1280,2560]:
        optimizer.grid_search(num_of_patients = num_of_patients, width = width, height = height, num_of_fire_departments=num_of_fire_departments, num_of_rendezvous_points=num_of_rendezvous_points, num_of_basehospitals=num_of_basehospitals, map_relocations=10, qa_trial_count=5, use_d_wave=True, is_new_algorithm_p1 = True, is_new_algorithm_p2 = True)

if __name__ == "__main__":
    main()

