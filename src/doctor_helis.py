# -*- coding: utf-8 -*-
import optimizer
import animation as ani
from matplotlib import animation

def main():
    num_of_fire_departments = 9 #救急車
    num_of_basehospitals = 8 #ドクターヘリ
    width = 20000
    height = 20000
    animes = []
    params = []

    cnt = 0
    for num_of_patients in [8,]:
      for num_of_rendezvous_points in [11,]:
        map_relocations=10
        qa_trial_count=3
        params.append([cnt,width,height,num_of_fire_departments,num_of_patients,\
                       num_of_rendezvous_points,num_of_basehospitals,\
                       map_relocations,qa_trial_count])  
        cnt += 1
        df, best_world_classic, best_world_ip, best_world_qa = optimizer.evaluate(num_of_patients = num_of_patients,\
                width = width, height = height, num_of_fire_departments=num_of_fire_departments,\
                num_of_rendezvous_points=num_of_rendezvous_points, num_of_basehospitals=num_of_basehospitals,\
                map_relocations=map_relocations, qa_trial_count=qa_trial_count, use_d_wave=True)
                #is_new_algorithm_p1 = False, is_new_algorithm_p2 = False)

        print(df)

        anim = ani.Animation(best_world_classic, best_world_ip, best_world_qa, 1000).animate()
        animes.append(anim)

    for anime, params in zip(animes,params):
        video_filename = '{:02}'.format(params[0]) + '_' + 'w' + str(params[1]) + '_' + 'h' + str(params[2]) + \
                    '_' +'a' + str(params[3]) + '_' + 'p' + str(params[4]) + \
                    '_' +'r' + str(params[5]) + '_' + 'd' + str(params[6]) + \
                    '_' +'m' + str(params[7]) + '_' + 'q' + str(params[8]) + '.avi'
        print(video_filename)
        writervideo = animation.FFMpegWriter(fps=30)
        anime.save(video_filename, writer=writervideo)
        print('Done')

if __name__ == "__main__":
    main()

