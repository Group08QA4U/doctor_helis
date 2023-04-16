# -*- coding: utf-8 -*-
import optimizer
import animation as ani
from matplotlib import animation

import parameters
import argparse

parser = argparse.ArgumentParser(description='Parameters of virtual map.')
parser.add_argument('-m', dest='map_realocation_cnt', type=int, default=10, help='map realocation count')
parser.add_argument('-q', dest='qa_retry_cnt', type=int, default=3, help='QA retry count')
parser.add_argument('-width', dest='width', type=int, default=20000, help='width')
parser.add_argument('-height', dest='height', type=int, default=20000, help='height')
parser.add_argument('-p', dest='patients', type=int, default=8, help='num of patients')
parser.add_argument('-a', dest='ambulance', type=int, default=9, help='num of ambulances')
parser.add_argument('-r', dest='rendezvous_points', type=int, default=11, help='num of rendezvous points')
parser.add_argument('-d', dest='doctorhelis', type=int, default=8, help='num of doctor helis')
parser.add_argument('-l1', dest='lamda1', type=float, default=8, help='hayper parameter for Q1')
parser.add_argument('-l2', dest='lamda2', type=float, default=8, help='hayper parameter for Q2')
parser.add_argument('-l3', dest='lamda3', type=float, default=8, help='hayper parameter for Q3')



def main(map_realocation_cnt, qa_trial_count, width, height, num_of_patients, num_of_fire_departments, num_of_rendezvous_points, num_of_basehospitals, lams):

    params = [width,height,num_of_patients,num_of_fire_departments,\
              num_of_rendezvous_points,num_of_basehospitals,\
              map_realocation_cnt,qa_trial_count,lams]

    df, best_world_classic, best_world_ip, best_world_qa = optimizer.evaluate(num_of_patients = num_of_patients,\
                                                                              width = width, height = height,\
                                                                              num_of_fire_departments=num_of_fire_departments,\
                                                                              num_of_rendezvous_points=num_of_rendezvous_points,\
                                                                              num_of_basehospitals=num_of_basehospitals,\
                                                                              map_relocations=map_realocation_cnt,\
                                                                              qa_trial_count=qa_trial_count,\
                                                                              use_d_wave=True,\
                                                                              lams = lams)
    print(df)

    anime = ani.Animation(best_world_classic, best_world_ip, best_world_qa, 1000).animate()

    video_filename = 'w' + str(params[0]) + '_' + 'h' + str(params[1]) + \
                '_' +'p' + str(params[2]) + '_' + 'a' + str(params[3]) + \
                '_' +'r' + str(params[4]) + '_' + 'd' + str(params[5]) + \
                '_' +'m' + str(params[6]) + '_' + 'q' + str(params[7]) + \
                '_' + 'lams_' + ".".join([str(_) for _ in lams])  + '.avi'

    print(video_filename)
    writervideo = animation.FFMpegWriter(fps=30)
    anime.save(video_filename, writer=writervideo)
    print('Done')

if __name__ == "__main__":
    params = parser.parse_args()
    m = params.map_realocation_cnt
    q = params.qa_retry_cnt
    w = params.width
    h = params.height
    p = params.patients
    a = params.ambulance
    r = params.rendezvous_points
    d = params.doctorhelis
    l1 = params.lamda1
    l2 = params.lamda2
    l3 = params.lamda3
    lams = [l1, l2, l3]
    main(m, q, w, h, p, a, r, d, lams)

