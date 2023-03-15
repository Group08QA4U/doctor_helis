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
#parser.add_argument('-csv_file', dest='csv_file', type=str, help='target data', required=True)
#parser.add_argument('-save_dir', dest='save_dir', default='output', type=str, help='Directory for output files')


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
    return params_str, params_dict

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
    #print('Results of grid search for', APP_NAME)
    #for result in sorted(results, reverse=True):
    #    acc, params_str, params_dict = result
    #    result_dict = {}
    #    result_dict.update({'acc':acc})
    #    result_dict.update(params_dict)
    #    result_data.append(result_dict)
    #    print(acc, params_str)


    #result_csv_filename = os.path.join(save_dir, 'results_grid_search_' + APP_NAME + '.csv')
    #write_result_csv(result_csv_filename, result_data)
    #print('Save result csv to ',result_csv_filename)


if __name__ == '__main__':
    params = parser.parse_args()

    n_cpu = params.cpu

    life_saving_resources_params = parameters.LifeSavingResourcesParams()
    hyper_params = parameters.HyperParams()

    try:
        main(life_saving_resources_params)
    except KeyboardInterrupt:
        print('closeing...')
    exit()
