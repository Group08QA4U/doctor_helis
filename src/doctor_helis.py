# -*- coding: utf-8 -*-
import optimizer
import animation

def main():
    num_of_fire_departments = 9 #救急車
    num_of_basehospitals = 8 #ドクターヘリ
    width = 20000
    height = 20000

    for num_of_patients in [8,]:
      for num_of_rendezvous_points in [11,]:
        df, best_world_classic, best_world_ip, best_world_qa = optimizer.evaluate(num_of_patients = num_of_patients, width = width, height = height, num_of_fire_departments=num_of_fire_departments, num_of_rendezvous_points=num_of_rendezvous_points, num_of_basehospitals=num_of_basehospitals, map_relocations=10, qa_trial_count=5, use_d_wave=True, is_new_algorithm_p1 = True, is_new_algorithm_p2 = True)
        print(df)

        #anim = animation.Animation(best_world_classic, best_world_qa, 2000).animate()
        #anim

if __name__ == "__main__":
    main()

