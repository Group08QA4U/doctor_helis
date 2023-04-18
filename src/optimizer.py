#-*- coding: utf-8 -*-

import sys
import numpy as np
import time

import worlds

# 最適化ベースクラス
class Optimizer:
  def __init__(self):
    return

  def getCandidateRoutes(self, time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients, is_debug=False):

    self.num_of_patients = time_a2p.shape[1]
    self.num_of_ambulances = time_a2p.shape[0]
    self.num_of_rendezvous_points = time_p2r.shape[1]
    self.num_of_doctor_helis = time_r2d.shape[1]
    #self.num_of_sdf_helis = time_s2p.shape[1]
    self.num_of_basehospitals = time_r2h.shape[1]

    self.time_a2p = time_a2p
    self.time_p2r = time_p2r
    #self.time_s2p = time_s2p
    #self.time_p2h = time_p2h
    self.time_r2d = time_r2d
    self.time_r2h = time_r2h

    self.remaining_time_all_patients = remaining_time_all_patients
    
    if is_debug:
      print('num of patients',self.num_of_patients)
      print('num of ambulances',self.num_of_ambulances)
      print('num of rendezvous_points',self.num_of_rendezvous_points)
      print('num of doctor_helis',self.num_of_doctor_helis)
      print('num of basehospitals',self.num_of_basehospitals)
          
      print('Estimate time from ambulance to patient (ambulance x patient)\n',self.time_a2p)
      print('Estimate time from patient to rendezvous_point (patient x rendezvous_point)\n',self.time_p2r)
      print('Estimate time from rendezvous_point to doctor heli (rendezvous_point x doctor heli)\n',self.time_r2d)
      print('Estimate time from rendezvous_point to basehospital (rendezvous_point x basehospital)\n',self.time_r2h)    

      print('remaining time for all patients [(FirstAid, Treatement, Tobasehospital),]\n',remaining_time_all_patients)


class Classic(Optimizer):
  def __init__(self):
    super().__init__()

  def getCandidateRoutes(self, time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients):
    super().getCandidateRoutes(time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients)
    
    candidate_routes = np.zeros((self.num_of_patients, self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_basehospitals ))
    reserved_ambulances = np.zeros(self.num_of_ambulances)
    reserved_rendezvous_points = np.zeros(self.num_of_rendezvous_points)
    reserved_doctor_helis = np.zeros(self.num_of_doctor_helis)
    #reserved_sdf_helis = np.zeros(self.num_of_sdf_helis)
    reserved_basehospitals = np.zeros(self.num_of_basehospitals)
    
    best_routes = []
    for i in range(self.num_of_patients):
      best_routes.append([i, [-1, -1, -1, -1 ], remaining_time_all_patients[i], -1, -remaining_time_all_patients[i]])
      #min_route1 = min_time_to_treatment_1 = min_time_to_treatment_2 = sys.maxsize
      min_time_to_treatment_1 = min_time_to_treatment_2 = sys.maxsize
      a2p = p2r = r2d = d2h = -1      
      #s2p = p2h = -1   
      for j in range(self.num_of_ambulances):
        if reserved_ambulances[j] == True:
          continue 
        for k in range(self.num_of_rendezvous_points):
          if reserved_rendezvous_points[k] == True:
            continue           
          for l in range(self.num_of_doctor_helis):
            if reserved_doctor_helis[l] == True:
              continue 

            time_to_treatment = max( time_a2p[j][i] + time_p2r[i][k] , time_r2d[k][l] )
            if min_time_to_treatment_1 > time_to_treatment:
              min_time_to_treatment_1 = time_to_treatment
              a2p = j
              p2r = k
              r2d = l
              d2h = l # ドクターヘリは、出動した基地病院へ戻る

            #for m in range(self.num_of_basehospitals):
            #  #route_estimated_time = max( time_a2p[j][i] + time_p2r[i][k] , time_r2d[k][l] ) + time_r2h[k][m]
            #  time_to_treatment = max( time_a2p[j][i] + time_p2r[i][k] , time_r2d[k][l] )
            #  #print(i,j,k,l,m,-1,-1,route_estimated_time)
            #  if min_time_to_treatment_1 > time_to_treatment:
            #    min_time_to_treatment_1 = time_to_treatment
            #    #min_route1 = route_estimated_time
            #    a2p = j
            #    p2r = k
            #    r2d = l
            #    d2h = m



      if min_time_to_treatment_1 < min_time_to_treatment_2:
        reserved_ambulances[a2p] = True
        reserved_rendezvous_points[p2r] = True
        reserved_doctor_helis[r2d] = True
        #[patient#, [a2p, p2r, r2d, d2h], Time left for the patient, Estimated time to start treatment, Score(Difference b/w the time left for the patient and the time to start treatment)]
        best_routes[i] = [i, [a2p, p2r, r2d, d2h ], remaining_time_all_patients[i], min_time_to_treatment_1, remaining_time_all_patients[i] - min_time_to_treatment_1]
        #best_routes.append([i, [a2p, p2r, r2d, d2h ], remaining_time_all_patients[i], min_time_to_treatment_1, remaining_time_all_patients[i] - min_time_to_treatment_1])




    return best_routes

# 整数計画Optimizerクラス
import pulp
import itertools

class IP(Optimizer):
  def __init__(self):
    super().__init__()

  def getCandidateRoutes(self, time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients):
    super().getCandidateRoutes(time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients)
    
    candidate_routes = np.zeros((self.num_of_patients, self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_basehospitals ))
    reserved_ambulances = np.zeros(self.num_of_ambulances)
    reserved_rendezvous_points = np.zeros(self.num_of_rendezvous_points)
    reserved_doctor_helis = np.zeros(self.num_of_doctor_helis)
    #reserved_sdf_helis = np.zeros(self.num_of_sdf_helis)
    reserved_basehospitals = np.zeros(self.num_of_basehospitals)

    # 定数読み込み
    N = self.num_of_patients
    A = self.num_of_ambulances
    R = self.num_of_rendezvous_points
    D = self.num_of_doctor_helis
    B = self.num_of_basehospitals
    M = A + R + D + B

    # 変数のアドレスリスト作成
    pr = list( itertools.product(range(A),range(N),range(R),range(D)) )  # 全体
    pra = list( itertools.product(range(A),range(N)) )
    prr = list( itertools.product(range(N),range(R)) )
    prd = list( itertools.product(range(R),range(D)) )

    # 決定変数定義
    xa = {(i,j):pulp.LpVariable('xa%d_%d'%(i,j), cat="Binary") for i,j in pra}    # 救急車
    xr = {(j,k):pulp.LpVariable('xr%d_%d'%(j,k), cat="Binary") for j,k in prr}    # ランデブーポイント
    xd = {(k,l):pulp.LpVariable('xd%d_%d'%(k,l), cat="Binary") for k,l in prd}    # ドクターヘリ

    # 非線形関数の線形化用バイナリ変数定義
    y = {(i,j,k,l):pulp.LpVariable('y%d_%d_%d_%d'%(i,j,k,l), cat="Binary") for i,j,k,l in pr}

    # 最適化モデルの定義
    mip_model = pulp.LpProblem(sense=pulp.LpMinimize)

    # Objective function -> minimize(搬送時間)
    objective =  pulp.lpSum(y[i,j,k,l] * max(time_a2p[i][j] + time_p2r[j][k], time_r2d[k][l]) for i,j,k,l in pr)

    mip_model += objective

    # Constraint functions
    ## Ambulance -> Patients
    for i in range(A):
        mip_model += pulp.lpSum(xa[i,j] for j in range(N)) <= 1
    for j in range(N):
        mip_model += pulp.lpSum(xa[i,j] for i in range(A)) == 1

    ## Patient -> Randezvous points
    for j in range(N):
        mip_model += pulp.lpSum(xr[j,k] for k in range(R)) == 1
    for k in range(R):
        mip_model += pulp.lpSum(xr[j,k] for j in range(N)) <= 1

    #mip_model += pulp.lpSum(xr[j,k] for j in range(N) for k in range(R)) == N

    ## Randezvous points -> Doctor helis
    for k in range(R):
        mip_model += pulp.lpSum(xd[k,l] for l in range(D)) <= 1
    for l in range(D):
        mip_model += pulp.lpSum(xd[k,l] for k in range(R)) <= 1

    #mip_model += pulp.lpSum(xd[k,l] for k in range(R) for l in range(D)) == N

    ## 非線形関数→線形関数のための制約条件
    for i in range(A):
        for j in range(N):
            for k in range(R):
                for l in range(D):
                    mip_model += 2 - (xa[i,j] + xr[j,k] + xd[k,l]) + y[i,j,k,l] >= 0
                    mip_model += xa[i,j] - y[i,j,k,l] >= 0
                    mip_model += xr[j,k] - y[i,j,k,l] >= 0
                    mip_model += xd[k,l] - y[i,j,k,l] >= 0

    for j in range(N):
        mip_model += pulp.lpSum(y[i,j,k,l] for i in range(A) for k in range(R) for l in range(D)) == 1

    # ソルバー設定
    #solver = pulp.PULP_CBC_CMD(threads=10, timeLimit=600)    # 4 thread 並列指定
    solver = pulp.GUROBI_CMD()

    print("start pulp solver :{}".format(solver))

    # ソルバー起動
    mip_model.solve(solver)

    # 実行可能解が存在したかを表示
    print(pulp.LpStatus[mip_model.status])

    # 目的関数の計算値
    print(pulp.value(mip_model.objective))

    print("Ambulance -> Patients")
    for i,x in xa.items():
        if pulp.value(x) > 0:
            print(i)

    print("Patient -> Rendezvous points")
    for i,x in xr.items():
        if pulp.value(x) > 0:
            print(i)

    print("Rendezvous point -> Doctor helis")
    for i,x in xd.items():
        if pulp.value(x) > 0:
            print(i)


    print("routes")

    # 結果の格納
    best_routes = []
    ans_num = 0
    for i,x_a2p in xa.items():
        if pulp.value(x_a2p) > 0:
            best_routes.append([i[1], [-1, -1, -1, -1 ], remaining_time_all_patients[i[1]], -1, -remaining_time_all_patients[i[1]]])
            a2p = i[0]
            for j,x_p2r in xr.items():
                if pulp.value(x_p2r) > 0:
                    if j[0] == i[1]:
                        p2r = j[1]
                        for k,x_r2d in xd.items():
                            if pulp.value(x_r2d) > 0:
                                if k[0] == p2r:
                                    r2d = k[1]
                                    d2h = k[1] # ドクターヘリは、出動した基地病院へ戻る
                                    reserved_ambulances[a2p] = True
                                    reserved_rendezvous_points[p2r] = True
                                    reserved_doctor_helis[r2d] = True
                                    time_to_treatment = max( time_a2p[i[0]][i[1]] + time_p2r[j[0]][j[1]] , time_r2d[k[0]][k[1]] )
                                    print(i[0])
                                    best_routes[ans_num] = [i[1], [a2p, p2r, r2d, d2h ], remaining_time_all_patients[i[1]], time_to_treatment, remaining_time_all_patients[i[1]] - time_to_treatment]
                                    print(best_routes[ans_num])
            ans_num += 1

    return best_routes

from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from openjij import SQASampler
# .env ファイルをロードして環境変数へ反映
import os
from dotenv import load_dotenv
load_dotenv()

# 環境変数を参照
DWAVE_TOKEN = os.getenv('DWAVE_TOKEN')
#print('DWAVE_TOKEN',DWAVE_TOKEN)

class QA(Optimizer):
  def __init__(self, area, use_d_wave = True, use_hybrid = True, is_new_algorithm_p1 = False, is_new_algorithm_p2 = False, is_max_algorithm_p3 = False, lams = None):
    super().__init__()
    self.token = DWAVE_TOKEN 
    self.endpoint = 'https://cloud.dwavesys.com/sapi/'
    self.area = area # 仮想地図面積
    self.use_d_wave = use_d_wave
    self.use_hybrid = use_hybrid
    self.is_new_algorithm_p1 = is_new_algorithm_p1
    self.is_new_algorithm_p2 = is_new_algorithm_p2
    self.is_max_algorithm_p3 = is_max_algorithm_p3

    self.lams = lams
    self.qubo = {}

  def getCandidateRoutes(self, time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients):
    super().getCandidateRoutes(time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients)
    # QUBOが空なら生成  
    if any(self.qubo) != True:
      num_route = self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_basehospitals
      route_estimated_time = []
      candidate_routes = []
      diff_time = np.zeros(self.num_of_patients*num_route).reshape(self.num_of_patients,num_route)
    
      # 生存時間、搬送時間の計算
      estimate_time = np.arange(self.num_of_patients * num_route).reshape((self.num_of_patients, self.num_of_ambulances, self.num_of_rendezvous_points, self.num_of_doctor_helis, self.num_of_basehospitals))
      route_to_start_treatment = self.num_of_patients * self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis
      estimate_time_to_start_treatment = np.arange(route_to_start_treatment).reshape((self.num_of_patients, self.num_of_ambulances, self.num_of_rendezvous_points, self.num_of_doctor_helis))
      for i in range(self.num_of_patients):
        for j in range(self.num_of_ambulances):
          for k in range(self.num_of_rendezvous_points):    
            for l in range(self.num_of_doctor_helis):
              for m in range(self.num_of_basehospitals):
                n_route =  j*self.num_of_rendezvous_points + k*self.num_of_doctor_helis + l*self.num_of_basehospitals + m
                estimate_time[i][j][k][l][m] = remaining_time_all_patients[i] - (max(time_a2p[j][i] + time_p2r[i][k], time_r2d[k][l]) + time_r2h[k][m])
                estimate_time_to_start_treatment[i][j][k][l] = max(time_a2p[j][i] + time_p2r[i][k], time_r2d[k][l])
      
      if self.lams != None:   
        self.qubo = self.QUBO(time_a2p, time_p2r, time_r2d, time_r2h, estimate_time, remaining_time_all_patients, lam1 = self.lams[0], lam2 = self.lams[1], lam3 = self.lams[2] )
      else:
        self.qubo = self.QUBO(time_a2p, time_p2r, time_r2d, time_r2h, estimate_time, remaining_time_all_patients, )
      
    self.Nsamples=10
      
    if self.use_d_wave:
        if self.use_hybrid:
          sampler = LeapHybridSampler(solver='hybrid_binary_quadratic_model_version2', token=self.token, endpoint=self.endpoint)
          results = sampler.sample_qubo(self.qubo)
        else:
          dw_sampler = DWaveSampler(solver='Advantage_system1.1', token=self.token, endpoint=self.endpoint)
          sampler = EmbeddingComposite(dw_sampler)
          results = sampler.sample_qubo(self.qubo, num_reads=self.Nsamples)

    else:      
      # OpenJIJ
      sampler = SQASampler(num_sweeps = 3000)
      results = sampler.sample_qubo(self.qubo, num_reads=self.Nsamples)
      

    return self.postProcessing(results, time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients)


  def postProcessing(self, results, time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients):
    # D-waveの結果を使用しやすいデータに加工する
    N = self.num_of_patients
    a = self.num_of_ambulances
    r = self.num_of_rendezvous_points
    d = self.num_of_doctor_helis
    b = self.num_of_basehospitals

    #print(results)
    #print(results.record[0][0].reshape(N,a+r+d+b))


    for i, result in enumerate(results.record):
      print('   opt#',i)
      is_found_opt_routes = True
      ambulance_cnt= 0
      rendezvous_point_cnt= 0
      doctor_helis_cnt= 0
      basehospital_cnt= 0
      opt_routes = []
      #for patient, resources in enumerate(result[0].reshape(N,a+r+d+b)):
      for patient, resources in enumerate(result[0].reshape(N,a+r+d)):

        #min_time_to_treatment = max(time_a2p[ambulance_num][patient] + time_p2r[patient][rendezvous_point_num], time_r2d[rendezvous_point_num][doctor_heli_num])
        #score = remaining_time_all_patients[patient] - min_time_to_treatment
        opt_routes.append([patient,[-1, -1, -1, -1], remaining_time_all_patients[patient], -1, -remaining_time_all_patients[patient]])

        ambulance = resources[:a]
        rendezvous_point = resources[a:a+r]
        doctor_helis = resources[a+r:a+r+d]
        basehospital = resources[a+r+d:a+r+d+b]

        IS_FAILED = False
        try:
          ambulance_num = np.where(ambulance==1)[0][0]
        except:
          #print('    Fail #',patient,'No ambulance is chosen')
          IS_FAILED = True

        try:
          rendezvous_point_num = np.where(rendezvous_point==1)[0][0]
        except:
          #print('    Fail #',patient,'No rendezvous_point is chosen')
          IS_FAILED = True

        try:
          doctor_heli_num = np.where(doctor_helis==1)[0][0]
          # ドクターヘリは出動した基地病院へ帰還する。
          basehospital_num = doctor_heli_num
        except:
          #print('    Fail #',patient,'No doctor_helis is chosen')
          IS_FAILED = True

        #try:
        #  basehospital_num = np.where(basehospital==1)[0][0]
        #except:
        #  print('    Fail #',patient,'No basehospital is chosen')
        #  IS_FAILED = True

        if sum(ambulance) > 1:
          #print('    Fail #',patient,'Multiple ambulances are chosen:',ambulance)
          IS_FAILED = True

        if sum(rendezvous_point) > 1:
          #print('    Fail #',patient,'Multiple rendezvous_points are chosen:',rendezvous_point)
          IS_FAILED = True     

        if sum(doctor_helis) > 1:
          #print('    Fail #',patient,'Multiple doctor_helis are chosen:',doctor_helis)
          IS_FAILED = True

        if sum(basehospital) > 1:
          #print('    Fail #',patient,'Multiple basehospitals are chosen:',basehospital)
          IS_FAILED = True

        if IS_FAILED == True:
          print('    Fail #', patient, 'ambulance',ambulance, 'rendezvous_point',rendezvous_point, 'doctor_heli',doctor_helis, 'basehospital',basehospital)
          is_found_opt_routes = False
          continue

        
        if (ambulance_cnt + 1) > a:
          print('    Fail #', patient, 'ambulance',ambulance, 'rendezvous_point',rendezvous_point, 'doctor_heli',doctor_helis, 'basehospital',basehospital)
          #print('    Fail #',patient,'ambulances should be', a ,'or less:',ambulance)
          break


        if (rendezvous_point_cnt + 1) > r:
          print('    Fail #', patient, 'ambulance',ambulance, 'rendezvous_point',rendezvous_point, 'doctor_heli',doctor_helis, 'basehospital',basehospital)
          #print('    Fail #',patient,'rendezvous_point should be', r ,'or less:',rendezvous_point)
          break

        if (doctor_helis_cnt + 1) > d:
          print('    Fail #', patient, 'ambulance',ambulance, 'rendezvous_point',rendezvous_point, 'doctor_heli',doctor_helis, 'basehospital',basehospital)
          #print('    Fail #',patient,'doctor_helis should be', d ,'or less:',doctor_helis)
          break

        ambulance_cnt += 1
        rendezvous_point_cnt += 1
        doctor_helis_cnt += 1

        # 基地病院の使用数は制限しない
        #if (basehospital_cnt + 1) > b:
        #  print('    Fail #',patient,'basehospital should be', b ,'or less:',basehospital)
        #  break
        #basehospital_cnt += 1

        print('    OK   #', patient, 'ambulance',ambulance, 'rendezvous_point',rendezvous_point, 'doctor_heli',doctor_helis, 'basehospital',basehospital)
        min_route = (max(time_a2p[ambulance_num][patient] + time_p2r[patient][rendezvous_point_num], time_r2d[rendezvous_point_num][doctor_heli_num]) + time_r2h[rendezvous_point_num][basehospital_num])
        min_time_to_treatment = max(time_a2p[ambulance_num][patient] + time_p2r[patient][rendezvous_point_num], time_r2d[rendezvous_point_num][doctor_heli_num])
        score = remaining_time_all_patients[patient] - min_time_to_treatment
        # [patient#, [a2p, p2r, r2d, d2h], Time left for the patient, Estimated time to start treatment, Score(Difference b/w the time left for the patient and the time to start treatment)]
        #opt_routes.append([patient,[ambulance_num, rendezvous_point_num, doctor_heli_num, basehospital_num], remaining_time_all_patients[patient], min_time_to_treatment, score])
        #opt_routes[patient] = [patient,[ambulance_num, rendezvous_point_num, doctor_heli_num, basehospital_num], remaining_time_all_patients[patient], min_time_to_treatment, score]
        opt_routes[patient] = [patient,[ambulance_num, rendezvous_point_num, doctor_heli_num, doctor_heli_num], remaining_time_all_patients[patient], min_time_to_treatment, score]
      if is_found_opt_routes:
        break

    #if self.DEBUG:
    total_score = 0
    for route in opt_routes:
      if route[1][0] == -1 or route[1][1] == -1 or route[1][2] == -1 or route[1][3] == -1:
        total_score = None
        break
      total_score += route[3] # min_time_to_treatment
      #print(route)
    print('Total score:',total_score)

    return opt_routes


  def QUBO(self, time_a2p, time_p2r, time_r2d, time_r2h, estimate_time, remaining_time_all_patients, lam1=40.0, lam2=40.0, lam3 = -0.229):
    # Make QUBO from time_a2p, time_p2r... and remaining_time_all_patients
    Q = {}
    N = self.num_of_patients
    a = self.num_of_ambulances
    r = self.num_of_rendezvous_points
    d = self.num_of_doctor_helis
    b = self.num_of_basehospitals
    M = a + r + d + b

    #S = np.sqrt(self.area * 2)
    # 仮想地図の面積 x 単位面積当たりの2点間の平均距離
    #S = self.area * ((2 + np.sqrt(2) + 5 * np.log(np.sqrt(2)+1)) / 15)
    S = 1



    Q1 = {}
    # 要救助者は、各救命リソースを１つだけ選択するよう制限する
    for i in range(N):
      # 救急車
      for j in range(a):
        for k in range(a):
          Q1[(i * M + j, i * M + k)] = lam1
          if j == k:
            if self.is_new_algorithm_p1:
              Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1 * min(N,a,r,d))/a
            else:
              Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1)

      # ランデブーポイント
      for j in range(a, a + r):
        for k in range(a, a + r):
          Q1[(i * M + j, i * M + k)] = lam1
          if j == k:
            if self.is_new_algorithm_p1:            
              Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1 * min(N,a,r,d))/r
            else:
              Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1)

      # ドクターヘリ
      for j in range(a + r, a + r + d):
        for k in range(a + r, a + r + d):
          Q1[(i * M + j, i * M + k)] = lam1
          if j == k:
            if self.is_new_algorithm_p1:   
              Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1 * min(N,a,r,d))/d
            else:
              Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1)

      # 基地病院（基地病院の使用数は制限しない）
      #for j in range(a + r + d, a + r + d + b):
      #  for k in range(a + r + d, a + r + d + b):
      #    Q1[(i * M + j, i * M + k)] = lam1
      #    if j == k:
      #      if self.is_new_algorithm_p1:             
      #        Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1 * min(N,a,r,d))/b
      #      else:
      #        Q1[(i * M + j, i * M + k)] = Q1[(i * M + j, i * M + k)] - (2 * lam1)

    # 各救命リソースの数を制限する
    Q2 = {}
    # 救急車           
    for j in range(a):
      for i in range(N):
        for k in range(N):
          Q2[(i * M + j, k * M + j)] = lam2
          if i == k:
            if self.is_new_algorithm_p2:            
              Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] -(2 * lam2 * min(N,a,r,d))/a
            else:
              Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] -(2 * lam2)
    # ランデブーポイント
    for j in range(a, a + r):
      for i in range(N):
        for k in range(N):
          Q2[(i * M + j, k * M + j)] = lam2
          if i == k:
            if self.is_new_algorithm_p2:            
              Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] -(2 * lam2 * min(N,a,r,d))/r
            else:
              Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] -(2 * lam2)

    # ドクターヘリ
    for j in range(a + r, a + r + d):
      for i in range(N):
        for k in range(N):
          Q2[(i * M + j, k * M + j)] = lam2
          if i == k:
            if self.is_new_algorithm_p2:            
              Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] -(2 * lam2 * min(N,a,r,d))/d
            else:
              Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] -(2 * lam2)

    # 基地病院（基地病院の使用数は制限しない）
    #for j in range(a + r + d, a + r + d + b):
    #  for i in range(N):
    #    for k in range(N):
    #      Q2[(i * M + j, k * M + j)] = lam2
    #      if i == k:
    #       if self.is_new_algorithm_p2:
    #         Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] - (2 * lam2 * min(N,a,r,d))/b
    #       else:
    #         Q2[(i * M + j, k * M + j)] = Q2[(i * M + j, k * M + j)] - (2 * lam2)


    # 要救助者に残された時間と搬送時間を勘案する。
    Q3 = {}
    if self.use_d_wave != True:
      lam3 = -0.000001

    if self.is_max_algorithm_p3 == False:  
      #print('self.is_max_algorithm_p3',self.is_max_algorithm_p3)
      #exit()
      # 要救助者
      for i in range(N):

        # 救急車
        for j in range(a):

          # ランデブーポイント
          for k in range(a, a + r):
            # 救急車→要救助者→ランデブーポイント
            time_a2p2r = (time_a2p[j][i] + time_p2r[i][k-a])

            #Q3[(i * M + j, i * M + k)] = lam3 * (remaining_time_all_patients[i] - time_a2p2r)
            #Q3[(i * M + k, i * M + j)] = lam3 * (remaining_time_all_patients[i] - time_a2p2r)
            Q3[(i * M + j, i * M + k)] = lam3 * time_a2p2r / N / S
            Q3[(i * M + k, i * M + j)] = lam3 * time_a2p2r / N / S

        # ドクターヘリ
        for j in range(a + r, a + r + d):
          # ランデブーポイント
          for l in range(a, a + r):
            #Q3[(i * M + j, i * M + l)] = lam3 * (remaining_time_all_patients[i] - time_r2d[l-a][j-(a + r)])
            #Q3[(i * M + l, i * M + j)] = lam3 * (remaining_time_all_patients[i] - time_r2d[l-a][j-(a + r)])
            Q3[(i * M + j, i * M + l)] = lam3 * time_r2d[l-a][j-(a + r)] / N / S
            Q3[(i * M + l, i * M + j)] = lam3 * time_r2d[l-a][j-(a + r)] / N / S
    else:
      # 要救助者
      for i in range(N):

        # 救急車
        for j in range(a):

          # ランデブーポイント
          for k in range(a, a + r):
            # 救急車→要救助者→ランデブーポイント
            time_a2p2r = (time_a2p[j][i] + time_p2r[i][k-a])

            # ドクターヘリ
            for l in range(a + r, a + r + d):
              Q3[(i * M + j, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[k-a][l-(a + r)]) / N / S
              Q3[(i * M + k, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[k-a][l-(a + r)]) / N / S
            
              Q3[(i * M + j, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[k-a][l-(a + r)]) / N / S
              Q3[(i * M + l, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[k-a][l-(a + r)]) / N / S
            
              Q3[(i * M + k, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[k-a][l-(a + r)]) / N / S
              Q3[(i * M + l, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[k-a][l-(a + r)]) / N / S

          # ドクターヘリ
          for k in range(a + r, a + r + d):

            # ランデブーポイント
            for l in range(a, a + r):
              # 救急車→要救助者→ランデブーポイント
              time_a2p2r = (time_a2p[j][i] + time_p2r[i][l-a])

              Q3[(i * M + j, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[l-a][k-(a + r)]) / N / S
              Q3[(i * M + k, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[l-a][k-(a + r)]) / N / S
            
              Q3[(i * M + j, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[l-a][k-(a + r)]) / N / S
              Q3[(i * M + l, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[l-a][k-(a + r)]) / N / S
            
              Q3[(i * M + k, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[l-a][k-(a + r)]) / N / S
              Q3[(i * M + l, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[l-a][k-(a + r)]) / N / S
            

        # ランデブーポイント
        for j in range(a, a + r):

          # 救急車
          for k in range(a):
            # 救急車→要救助者→ランデブーポイント
            time_a2p2r = (time_a2p[k][i] + time_p2r[i][j-a])

            # ドクターヘリ
            for l in range(a + r, a + r + d):
              Q3[(i * M + j, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[j-a][l-(a + r)]) / N / S
              Q3[(i * M + k, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[j-a][l-(a + r)]) / N / S
            
              Q3[(i * M + j, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[j-a][l-(a + r)]) / N / S
              Q3[(i * M + l, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[j-a][l-(a + r)]) / N / S
            
              Q3[(i * M + k, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[j-a][l-(a + r)]) / N / S
              Q3[(i * M + l, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[j-a][l-(a + r)]) / N / S
            

          # ドクターヘリ
          for k in range(a + r, a + r + d):

            # 救急車
            for l in range(a):
              # 救急車→要救助者→ランデブーポイント
              time_a2p2r = (time_a2p[l][i] + time_p2r[i][j-a])            
              Q3[(i * M + j, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[j-a][k-(a + r)]) / N / S
              Q3[(i * M + k, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[j-a][k-(a + r)]) / N / S
            
              Q3[(i * M + j, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[j-a][k-(a + r)]) / N / S
              Q3[(i * M + l, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[j-a][k-(a + r)]) / N / S
            
              Q3[(i * M + k, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[j-a][k-(a + r)]) / N / S
              Q3[(i * M + l, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[j-a][k-(a + r)]) / N / S
            

        # ドクターヘリ
        for j in range(a + r, a + r + d):

          # 救急車
          for k in range(a):

            # ランデブーポイント
            for l in range(a, a + r):
              # 救急車→要救助者→ランデブーポイント
              time_a2p2r = (time_a2p[k][i] + time_p2r[i][l-a])            
              Q3[(i * M + j, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[l-a][j-(a + r)]) / N / S
              Q3[(i * M + k, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[l-a][j-(a + r)]) / N / S
           
              Q3[(i * M + j, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[l-a][j-(a + r)]) / N / S
              Q3[(i * M + l, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[l-a][j-(a + r)]) / N / S
            
              Q3[(i * M + k, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[l-a][j-(a + r)]) / N / S
              Q3[(i * M + l, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[l-a][j-(a + r)]) / N / S


          # ランデブーポイント
          for k in range(a, a + r):

            # 救急車
            for l in range(a):
              # 救急車→要救助者→ランデブーポイント
              time_a2p2r = (time_a2p[l][i] + time_p2r[i][k-a])            
              Q3[(i * M + j, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[k-a][j-(a + r)]) / N / S
              Q3[(i * M + k, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[k-a][j-(a + r)]) / N / S
           
              Q3[(i * M + j, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[k-a][j-(a + r)]) / N / S
              Q3[(i * M + l, i * M + j)] = lam3 * max(time_a2p2r, time_r2d[k-a][j-(a + r)]) / N / S
            
              Q3[(i * M + k, i * M + l)] = lam3 * max(time_a2p2r, time_r2d[k-a][j-(a + r)]) / N / S
              Q3[(i * M + l, i * M + k)] = lam3 * max(time_a2p2r, time_r2d[k-a][j-(a + r)]) / N / S

    # QUBO行列を可視化するvisualize_qubo.py入力するCSVを書き出してプログラムを終了する（debug）
    #saveQUBO(Q1,Q2,Q3)
    Q.update(Q1) 
    Q.update(Q2)     
    Q.update(Q3)
    return Q

import csv
def saveQUBO(Q1,Q2,Q3):
    qubos = [Q1,Q2,Q3]
    csvfiles = ['Q1.csv','Q2.csv','Q3.csv']

    for qubo, csv_file in zip(qubos, csvfiles): 

        csv_data = []
        for k, v in qubo.items():
            csv_data.append([k[0],k[1],v])
 
        print('Save as',csv_file)
        with open(csv_file, 'w', ) as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

    print('Done')



    exit()


import pandas as pd
import matplotlib.pyplot as plt



def output_as_csv(best_classic_total_scores_list, best_ip_total_scores_list, best_qa_total_scores_list, classic_processing_time_list, ip_processing_time_list, qa_processing_time_list, num_of_patients, num_of_fire_departments, num_of_rendezvous_points, num_of_basehospitals): 
  #title = 'patients=' + str(num_of_patients) + '_' + 'ambulance=' + str(num_of_fire_departments) + '_' + 'rendezvous_points=' + str(num_of_rendezvous_points) + '_' + 'doctor_helis=' + str(num_of_basehospitals)

  data = []
  cnt = 0
  for gr, ip, qa, gr_time, ip_time, qa_time  in zip(best_classic_total_scores_list, best_ip_total_scores_list, best_qa_total_scores_list, classic_processing_time_list, ip_processing_time_list, qa_processing_time_list):
    qa_max = np.max([x for x in qa if x]) if len([x for x in qa if x]) != 0 else 'None'
    print('cnt,num_of_patients,gr,gr_time,ip,ip_time,qa_max,qa_time',cnt,num_of_patients,gr,gr_time,ip,ip_time,qa_max,qa_time)
    data.append([cnt, (gr * 60)/num_of_patients, gr_time, (ip * 60)/num_of_patients, ip_time, (qa_max * 60)/num_of_patients if qa_max != 'None' else 'Not found', qa_time] )
    cnt += 1

  columns = ['map#','Greedy(score sec/patient)', 'Greedy(ptime sec)', 'Integer Programming(score sec/patient)', 'Integer Programming(ptime sec)', 'Quantum annealing(score sec/patient)', 'Quantum annealing(ptime sec)' ]
  df = pd.DataFrame(data, columns=columns)
  df.set_index("map#",inplace=True)
  print('df',df)
  pd.set_option('display.max_columns', None)
  pd.set_option('display.max_rows', None)


  #print(title)
  #print(df)

  #df.head(5)
  #return df
  #fig, ax =plt.subplots(1,1)
  #ax.table(cellText=data,colLabels=columns)

  #plt.show()

  #df.plot(title = title)

  # CSV ファイル
  ##file_name = 'evaluation_results_' + title + '.csv'
  file_name = 'evaluation_results.csv'
  df.to_csv(file_name)
  #files.download(file_name)
  return df

import copy
def evaluate(num_of_patients, map_relocations=1, qa_trial_count=1, width = 86000, height = 86000, num_of_fire_departments = 8, num_of_rendezvous_points = 10, num_of_basehospitals = 8, use_d_wave=True, is_new_algorithm_p1 = False, is_new_algorithm_p2 = False, lams=[34.63320538704878, 49.15841176773455, 4.08550171630701]):

  classic_total_scores_list = []
  classic_processing_time_list=  []
  ip_total_scores_list = []
  ip_processing_time_list = []
  qa_total_scores_list = []
  qa_processing_time_list=  []

  best_qa_animation_score = -100000
  best_world_base = None
  best_qa = None
  best_classic = None
  best_location = None

  for j in range(map_relocations):
    # 要救助者、救命リソースをランダム配置
    world_base = worlds.World(width = width, height = height, num_of_patients=num_of_patients, num_of_fire_departments = num_of_fire_departments, num_of_rendezvous_points = num_of_rendezvous_points, num_of_basehospitals = num_of_basehospitals)

    # 古典コンピューターで計算
    start = time.time()
    world_classic = copy.deepcopy(world_base)
    classic = Classic()
    best_classic = copy.deepcopy(classic)
    classic_total_score = world_classic.getTotalScore(classic)
    classic_total_scores_list.append(classic_total_score)
    classic_processing_time_list.append( time.time() - start)

    # 整数計画で計算
    start = time.time()
    world_ip = copy.deepcopy(world_base)
    ip = IP()
    best_ip = copy.deepcopy(ip)
    ip_total_score = world_ip.getTotalScore(ip)
    ip_total_scores_list.append(ip_total_score)
    ip_processing_time_list.append( time.time() - start)
    #start = time.time()
    #world_ip = copy.deepcopy(world_base)
    #ip = Classic()
    #best_ip = copy.deepcopy(ip)
    #ip_total_score = world_ip.getTotalScore(ip)
    #ip_total_scores_list.append(ip_total_score)
    #ip_processing_time_list.append( time.time() - start)

    # QAで計算
    start = time.time()
    qa_total_scores = []
    #lams=[39.0,39.0,2.5]
    #lam3 = 4.08550171630701 / ((2 + np.sqrt(2) + 5 * np.log(np.sqrt(2)+1)) / 15)
    #lams=[34.63320538704878, 49.15841176773455, lam3]

    #lams=[38.2604549772066, 40.227539295627075, 3.611067604728468]
    #lams=[34.63320538704878, 49.15841176773455, 4.08550171630701]
    is_new_algo = False
    is_max_algo = False

    # QA
    qa = QA( width * height, use_d_wave=use_d_wave, is_new_algorithm_p1 = is_new_algo, is_new_algorithm_p2 = is_new_algo, is_max_algorithm_p3=is_max_algo, lams=lams)
    for k in range(qa_trial_count):
      title = 'patients#:' + str(num_of_patients) + ' ' + 'relocation#:' + str(j) + ' '  + \
              'qa_trial_count#:' + str(k) + ' ' + 'ambulance:' + str(num_of_fire_departments) + ' ' + \
              'rendezvous_points:' + str(num_of_rendezvous_points) + ' ' + \
              'doctor_helis:' + str(num_of_basehospitals) + ' lams:' + " ".join([str(_) for _ in lams]) + ' ' + \
              'is_new_algo:' + str(is_new_algo) + ' ' + 'is_max_algo:' + str(is_max_algo) 
      print(title)      
      world_qa = copy.deepcopy(world_base)


      qa_total_score = world_qa.getTotalScore(qa)
      qa_total_scores.append(qa_total_score)
 
      if qa_total_score != None:
        # 生成するアニメーションを決定するロジック
        # 従来手法とQAを比較して、QAが大きな差を付けて従来手法に勝る地図をアニメーションとして残す
        if best_qa_animation_score < (classic_total_score - qa_total_score):
          best_qa_animation_score = classic_total_score - qa_total_score
          best_qa = copy.deepcopy(qa)
          best_world_base = copy.deepcopy(world_base)
          best_location = j

    qa_total_scores_list.append(qa_total_scores)
    qa_processing_time_list.append( time.time() - start)



  df = output_as_csv(classic_total_scores_list, ip_total_scores_list, qa_total_scores_list, classic_processing_time_list, ip_processing_time_list, qa_processing_time_list, num_of_patients, num_of_fire_departments = num_of_fire_departments, num_of_rendezvous_points = num_of_rendezvous_points, num_of_basehospitals = num_of_basehospitals)

  print('# Classic despatch')
  world_classic = copy.deepcopy(best_world_base)
  world_classic.despatch(best_classic)  

  print('# IP despatch')
  world_ip = copy.deepcopy(best_world_base)
  world_ip.despatch(best_ip)  

  print('# QA despatch location#',best_location)
  world_qa = copy.deepcopy(best_world_base)
  world_qa.despatch(best_qa)  
  
  
  return df, world_classic, world_ip, world_qa

def grid_search(life_saving_resources_params, hyper_params, map_relocations=10, qa_trial_count=5, width = 86000, height = 86000, use_d_wave=True, is_new_algorithm_p1 = True, is_new_algorithm_p2 = True):

  num_of_patients = life_saving_resources_params.patients
  num_of_fire_departments = life_saving_resources_params.fire_departments
  num_of_rendezvous_points = life_saving_resources_params.rendezvous_points
  num_of_basehospitals = life_saving_resources_params.basehospitals


  qa_total_scores_list = []
  classic_total_scores_list = []

  best_qa_total_score = sys.maxsize
  best_world_base = None
  best_qa = None
  #best_classic = None
  best_location = None

  params_dict = {'width':width, 'height':height}

  for j in range(map_relocations):
    print('map_relocations',j)
    params_dict['map_relocations'] = j
    # 要救助者、救命リソースをランダム配置
    world_base = worlds.World(width = width, height = height, num_of_patients=num_of_patients, num_of_fire_departments = num_of_fire_departments, num_of_rendezvous_points = num_of_rendezvous_points, num_of_basehospitals = num_of_basehospitals)

    # QAで計算
    qa_total_scores = []
    for k in range(qa_trial_count):
      print('qa_trial_count',k)
      params_dict['qa_trial_count'] = k
      #title = '# of patients:' + str(num_of_patients) + ' ' + 'relocation#:' + str(j) + ' '  + 'qa_trial_count#:' + str(k) + ' ' + 'ambulance:' + str(num_of_fire_departments) + ' ' + 'rendezvous_points:' + str(num_of_rendezvous_points) + ' ' + 'doctor_helis:' + str(num_of_basehospitals)
      title_str, title_dict = life_saving_resources_params.get_title_from_params()
      params_dict.update(title_dict)
      #print(params_dict)      
      world_qa = copy.deepcopy(world_base)

      hyper_params.init_parameters()

      while hyper_params.set_next_params():
        lam1 = hyper_params.lam1
        lam2 = hyper_params.lam2
        lam3 = hyper_params.lam3
        _, dict_hyper_parameters = hyper_params.get_title_from_params()
        params_dict.update(dict_hyper_parameters)
      # QA
        #qa = QA(use_d_wave=use_d_wave, is_new_algorithm_p1 = is_new_algorithm_p1, is_new_algorithm_p2 = is_new_algorithm_p2, lams = [lam1,lam2,lam3])
        qa = QA( width * height, use_d_wave=use_d_wave, is_new_algorithm_p1 = is_new_algorithm_p1, is_new_algorithm_p2 = is_new_algorithm_p2, lams = [lam1,lam2,lam3])

        qa_total_score = world_qa.getTotalScore(qa)
        params_dict['score'] = qa_total_score
        print('Summary',params_dict)




def bayes(X, qa_trial_count, width, height, num_of_patients, num_of_fire_departments, num_of_rendezvous_points, num_of_basehospitals, use_d_wave=True, is_new_algo = False, is_max_algo = False ):
  lams=[X[0],X[0],X[1]]

  #map_relocations=1
  #qa_trial_count=2
  #width = 20000
  #height = 20000
  #num_of_patients = 14
  #num_of_fire_departments = 14
  #num_of_rendezvous_points = 20
  #num_of_basehospitals = 14

  #use_d_wave = True
  #is_new_algo = False
  #is_max_algo = True

  best_qa_total_score = sys.maxsize

  # 要救助者、救命リソースをランダム配置
  world_base = worlds.World(width = width, height = height, num_of_patients=num_of_patients, num_of_fire_departments = num_of_fire_departments, num_of_rendezvous_points = num_of_rendezvous_points, num_of_basehospitals = num_of_basehospitals)

  # QAで計算
  qa_total_scores = []
  qa = QA(width * height, use_d_wave=use_d_wave, is_new_algorithm_p1 = is_new_algo, is_new_algorithm_p2 = is_new_algo, is_max_algorithm_p3=is_max_algo, lams=lams)
  for k in range(qa_trial_count):
    title = 'patients#:' + str(num_of_patients) + ' ' + \
            'qa_trial_count#:' + str(k) + ' ' + 'ambulance:' + str(num_of_fire_departments) + ' ' + \
            'rendezvous_points:' + str(num_of_rendezvous_points) + ' ' + \
            'doctor_helis:' + str(num_of_basehospitals) + ' lams:' + " ".join([str(_) for _ in lams]) + ' ' + \
            'is_new_algo:' + str(is_new_algo) + ' ' + 'is_max_algo:' + str(is_max_algo)
    print(title)
    world_qa = copy.deepcopy(world_base)

    # QA
    #qa = QA(use_d_wave=use_d_wave, is_new_algorithm_p1 = is_new_algo, is_new_algorithm_p2 = is_new_algo, is_max_algorithm_p3=is_max_algo, lams=lams)

    qa_total_score = world_qa.getTotalScore(qa)
    qa_total_scores.append(qa_total_score)

    if qa_total_score != None:
      if best_qa_total_score > qa_total_score:
        best_qa_total_score = qa_total_score

  print('X, best_qa_total_score',X, best_qa_total_score)
  return X, best_qa_total_score
