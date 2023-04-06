# -*- coding: utf-8 -*-
import math
from numpy import linalg as LA
from matplotlib import animation, rc
#from IPython.display import HTML
import plotable
import numpy as np
import matplotlib.pyplot as plt

class World:
  def __init__(self, width, height, num_of_patients, num_of_rendezvous_points, num_of_basehospitals, num_of_fire_departments):

    #仮想地図の解像度（メートル）10km x 10km
    self.width = width
    self.height = height

    self.step = 0.0 # minuts
    self.time = 0

    self.num_of_patients = num_of_patients
    self.num_of_ambulances = self.num_of_fire_departments = num_of_fire_departments
    self.num_of_rendezvous_points = num_of_rendezvous_points
    self.num_of_doctor_helis = self.num_of_basehospitals = num_of_basehospitals
    #self.num_of_sdf_helis = self.num_of_sdf_bases = num_of_sdf_bases


    # 要救助者
    self.patients = []
    for _ in range(self.num_of_patients):
      self.patients.append(plotable.Patient(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 消防署
    self.fire_departments = []
    for _ in range(self.num_of_fire_departments):
      self.fire_departments.append(plotable.FireDepartment(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 救急車 
    self.ambulances = []
    for i in range(self.num_of_ambulances):
      x, y = self.fire_departments[i].getPosition()
      self.ambulances.append(plotable.Ambulance(x, y))      

    # ランデブーポイント
    self.rendezvous_points = []
    for _ in range(self.num_of_rendezvous_points):
      self.rendezvous_points.append(plotable.RendezvousPoint(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 基地病院
    self.basehospitals = []
    for _ in range(self.num_of_basehospitals):
      self.basehospitals.append(plotable.BaseHospital(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # ドクターヘリ
    self.doctor_helis = []
    for i in range(self.num_of_doctor_helis):
      x, y = self.basehospitals[i].getPosition() 
      self.doctor_helis.append(plotable.DrHeli(x, y))

    self.calcDistance4AllEdges()

  def initPlot(self, ax, title):
    plt.close()
    self.ax = ax
    self.title = title

    self.charts = []

    # 基地病院
    if len(self.basehospitals) > 0:
      c = self.basehospitals[0].getColor()
      m = self.basehospitals[0].getMarker()
      l = self.basehospitals[0].getLabel()
      self.point_basehospitals = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)    
      offset = []
      for basehospital in self.basehospitals:
        x, y = basehospital.getPosition()
        offset.append([x, y])
      self.charts.append(self.point_basehospitals)

    # 消防署
    if len(self.fire_departments) > 0:
      c = self.fire_departments[0].getColor()
      m = self.fire_departments[0].getMarker()
      l = self.fire_departments[0].getLabel()    
      self.point_fire_departments = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)    
      offset = []
      for fire_department in self.fire_departments:
        x, y = fire_department.getPosition()
        offset.append([x, y])
      self.charts.append(self.point_fire_departments)

    # ランデブーポイント
    if len(self.rendezvous_points) > 0:
      c = self.rendezvous_points[0].getColor()
      m = self.rendezvous_points[0].getMarker()
      l = self.rendezvous_points[0].getLabel()       
      self.point_rendezvous_points = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)    
      self.charts.append(self.point_rendezvous_points)
      offset = []
      for rendezvous_point in self.rendezvous_points:
        x, y = rendezvous_point.getPosition()
        offset.append([x, y])

    # 救急車
    if len(self.ambulances) > 0:
      c = self.ambulances[0].getColor()
      m = self.ambulances[0].getMarker()
      l = self.ambulances[0].getLabel()       
      self.route_ambulances = []
      self.point_ambulances = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)   
      self.charts.append(self.point_ambulances)    
      for ambulance in self.ambulances:
        line, = self.ax.plot([], [], c=c, linestyle="dashed")
        self.route_ambulances.append(line)
        self.charts.append(line)    
  
    # ドクターヘリ
    if len(self.doctor_helis) > 0:
      c = self.doctor_helis[0].getColor()
      m = self.doctor_helis[0].getMarker()
      l = self.doctor_helis[0].getLabel()      
      self.route_doctor_helis = []
      self.point_doctor_helis = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)   
      self.charts.append(self.point_doctor_helis)
      for doctor_heli in self.doctor_helis:
        line, = self.ax.plot([], [], c=c, linestyle="dashed")
        self.route_doctor_helis.append(line)
        self.charts.append(line)

    # 要救助者
    if len(self.patients) > 0:
      c = self.patients[0].getColor()
      m = self.patients[0].getMarker()
      l = self.patients[0].getLabel()  
      self.point_patients = self.ax.scatter([], [],  c=c, marker=m, label=l, s=100)   
      self.charts.append(self.point_patients)
      self.annotates = []
      for patient in self.patients:
        x, y = patient.getPosition()
        #first_aid, doctor_treatment, basehospital_treatment = patient.getRemainingTime()
        doctor_treatment = patient.getRemainingTime()
        fc='green'
        #if first_aid <= 5 or doctor_treatment <= 5 or basehospital_treatment <= 5:
        #  fc='yellow'
        #if first_aid <= 1 or doctor_treatment <= 1 or basehospital_treatment <= 0:
        #  fc='red'  
        #ano = self.ax.annotate("" ,xy=(patient.getPosition()[0], patient.getPosition()[1]), xytext=(-30, 30),textcoords='offset points', ha='left', va='top',bbox=dict(boxstyle='round,pad=0.5', fc=fc, alpha=0.3),arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0'))
        #ano = self.ax.annotate("" ,xy=(x, y), xytext=(-30, 30), textcoords='offset points', ha='left', va='top',bbox=dict(boxstyle='round,pad=0.5', fc=fc, alpha=0.3), arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0'))
        #self.annotates.append(ano)
        #self.charts.append(ano)
    
    # 凡例のスペースを空ける
    self.ax.set_xlim(0,self.width)
    self.ax.set_ylim(0,int(self.height*1.25))
    self.ax.legend(loc='upper center', ncol=3)
    self.ax.set_title(self.title)

    return tuple(self.charts)
    #return self.charts

  def plot(self):
    plt.close()

    # 基地病院
    if len(self.basehospitals):
      offset = []
      for basehospital in self.basehospitals:
        x, y = basehospital.getPosition()
        offset.append([x, y])
      self.point_basehospitals.set_offsets(offset)

    # 消防署
    if len(self.fire_departments):
      offset = []
      for fire_department in self.fire_departments:
        x, y = fire_department.getPosition()
        offset.append([x, y])
      self.point_fire_departments.set_offsets(offset)

    # ランデブーポイント
    if len(self.rendezvous_points):
      offset = []
      for rendezvous_point in self.rendezvous_points:
        x, y = rendezvous_point.getPosition()
        offset.append([x, y])
      self.point_rendezvous_points.set_offsets(offset)

    # 救急車
    if len(self.ambulances):
      offset = []
      for i, ambulance in enumerate(self.ambulances):
        x, y = ambulance.getPosition()
        offset.append([x, y])
        target_x, target_y = ambulance.getTargetPos()
        self.route_ambulances[i].set_data([x, target_x], [y, target_y])
      self.point_ambulances.set_offsets(offset)

    # ドクターヘリ
    if len(self.doctor_helis):
      offset = []
      for i, doctor_heli in enumerate(self.doctor_helis):
        x, y = doctor_heli.getPosition()
        offset.append([x, y])
        target_x, target_y = doctor_heli.getTargetPos()
        self.route_doctor_helis[i].set_data([x, target_x], [y, target_y])
      self.point_doctor_helis.set_offsets(offset)

    # 要救助者
    if len(self.patients):
      offset = []
      for i, patient in enumerate(self.patients):
        x, y = patient.getPosition()
        offset.append([x, y])
        doctor_treatment = patient.getRemainingTime()
        fc='green'
        #if first_aid <= 5 or doctor_treatment <= 5 or basehospital_treatment <= 5:
        #  fc='yellow'
        #if first_aid <= 1 or doctor_treatment <= 1 or basehospital_treatment <= 0:
        #  fc='red'  
        #self.annotates[i].set_position((x,y))
        #self.annotates[i].xy = (-30, 30)
        #self.annotates[i].set_text(str(int(first_aid)) + ', ' + str(int(doctor_treatment)) + ', ' + str(int(basehospital_treatment)))
        #self.annotates[i].set
        #self.charts.append(self.ax.annotate(str(int(first_aid)) + ', ' + str(int(doctor_treatment)) + ', ' + str(int(basehospital_treatment)) ,xy=(patient.getPosition()[0], patient.getPosition()[1]), xytext=(-30, 30),textcoords='offset points', ha='left', va='top',bbox=dict(boxstyle='round,pad=0.5', fc=fc, alpha=0.3),arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0')))
      
      self.point_patients.set_offsets(offset)

    return tuple(self.charts)
    #return self.charts


  def distance(self, x1, y1, x2, y2):
      d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
      return d

  def calcDistance4AllEdges(self):

    # ドクターヘリ速度（時速）
    if len(self.doctor_helis):    
      velocity_doctor_heli = self.doctor_helis[0].getVelocity()

    # 救急車（時速）
    if len(self.ambulances):
      velocity_ambulance = self.ambulances[0].getVelocity()

    # 救急車から要救助者への全ルートの距離を計算
    self.dist_a2p = np.zeros((self.num_of_ambulances, self.num_of_patients))
    self.time_a2p = np.zeros((self.num_of_ambulances, self.num_of_patients))
    for i, ambulance in enumerate(self.ambulances):
      x1 = ambulance.getPosition()[0]
      y1 = ambulance.getPosition()[1]
      for j, patient in enumerate(self.patients):
        x2 = patient.getPosition()[0]
        y2 = patient.getPosition()[1]
        self.dist_a2p[i][j] = self.distance(x1, y1, x2, y2)
        self.time_a2p[i][j] = (self.distance(x1, y1, x2, y2)*60)/(velocity_ambulance * 1000)

    # 要救助者 to ランデブーポイント
    self.dist_p2r = np.zeros((self.num_of_patients, self.num_of_rendezvous_points))
    self.time_p2r = np.zeros((self.num_of_patients, self.num_of_rendezvous_points))

    for i, patient in enumerate(self.patients):
      x1 = patient.getPosition()[0]
      y1 = patient.getPosition()[1]
      # 要救助者からランデブーポイントへの全ルートの距離を計算
      if len(self.rendezvous_points) and len(self.ambulances):
        for j, rendezvous_point in enumerate(self.rendezvous_points):
          x2 = rendezvous_point.getPosition()[0]
          y2 = rendezvous_point.getPosition()[1]
          self.dist_p2r[i][j] = self.distance(x1, y1, x2, y2)
          self.time_p2r[i][j] = (self.distance(x1, y1, x2, y2)*60)/(velocity_ambulance * 1000)

    # ランデブーポイント to ドクターヘリ
    self.dist_r2d = np.zeros((self.num_of_rendezvous_points, self.num_of_doctor_helis))
    self.time_r2d = np.zeros((self.num_of_rendezvous_points, self.num_of_doctor_helis))
    # ランデブーポイント to 基地病院
    self.dist_r2h = np.zeros((self.num_of_rendezvous_points, self.num_of_basehospitals))
    self.time_r2h = np.zeros((self.num_of_rendezvous_points, self.num_of_basehospitals))
    if len(self.rendezvous_points):    
      for i, rendezvous_point in enumerate(self.rendezvous_points):
        x1 = rendezvous_point.getPosition()[0]
        y1 = rendezvous_point.getPosition()[1]
        # ランデブーポイントからドクターヘリへの全ルートの距離を計算
        if len(self.doctor_helis):    
          for j, doctor_heli in enumerate(self.doctor_helis):
            x2 = doctor_heli.getPosition()[0]
            y2 = doctor_heli.getPosition()[1]
            self.dist_r2d[i][j] = self.distance(x1, y1, x2, y2)
            self.time_r2d[i][j] = (self.distance(x1, y1, x2, y2)*60)/(velocity_doctor_heli * 1000)
        # ランデブーポイントから基地病院への全ルートの距離を計算
        if len(self.basehospitals):           
          for k, basehospital in enumerate(self.basehospitals):
            x2 = basehospital.getPosition()[0]
            y2 = basehospital.getPosition()[1]
            self.dist_r2h[i][k] = self.distance(x1, y1, x2, y2)
            self.time_r2h[i][k] = (self.distance(x1, y1, x2, y2)*60)/(velocity_doctor_heli * 1000)

  def getEstimatedTimes4AllEdgesInMinutes(self):
    # time_a2p : 救急車から要救助者
    # time_p2r : 要救助者からランデブーポイント
    # time_s2p : 自衛隊ヘリから要救助者
    # time_r2h : ランデブーポイントから基地病院
    return self.time_a2p, self.time_p2r, self.time_r2d, self.time_r2h

  def getRemainingTime4AllPatients(self):
    remaining_time_patients = [] 
    for patient in self.patients:
      remaining_time_patients.append(patient.getRemainingTime())
    return remaining_time_patients


  # 時間変化に伴う要救助者に残された時間やヘリ等の位置を変化
  def update(self, i):

    # 救急車
    for i, ambulance in enumerate(self.ambulances):
      ambulance.update(self.step)

    # 要救助者
    for patient in self.patients:
      patient.update(self.step)

    # ランデブーポイント
    for rendezvous_point in self.rendezvous_points:
      rendezvous_point.update(self.step)

    # ドクターヘリ
    for i, doctor_heli in enumerate(self.doctor_helis):
      doctor_heli.update(self.step)

    # 基地病院
    for basehospital in self.basehospitals:
      basehospital.update(self.step)

    
    self.step = 0.1
    self.time += self.step
    return self.plot()

  def despatch(self, optimizer):
    time_a2p, time_p2r, time_r2d, time_r2h = self.getEstimatedTimes4AllEdgesInMinutes()
    remaining_time_all_patients = self.getRemainingTime4AllPatients()
    
    best_routes = optimizer.getCandidateRoutes(time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients)

    print('[patient#, [a2p, p2r, r2d, d2h], Time left for the patient, Estimated time to start treatment, Score(Difference b/w the time left for the patient and the time to start treatment)]')
    total_score = 0
    for route in best_routes:
      if route[1][0] == -1 or route[1][1] == -1 or route[1][2] == -1 or route[1][3] == -1:
        total_score = None
        break      
      total_score += route[3]
      print(route)
    print('Total score:',total_score)

    self.current_routes = best_routes
    # [patient#, [a2p, p2r, r2d, d2h], Time left for the patient, Estimated time to start treatment, Score(Difference b/w the time left for the patient and the time to start treatment)]
    for p, resources, _, _, _ in self.current_routes:
      if resources[1] != -1 and resources[0] != -1:
        self.ambulances[resources[0]].setTargets([self.patients[p], self.rendezvous_points[resources[1]], self.fire_departments[resources[0]]])

      if resources[2] != -1 and resources[1] != -1:        
        self.doctor_helis[resources[2]].setTargets([self.rendezvous_points[resources[1]], self.basehospitals[resources[2]]])

      if resources[1] != -1:
        self.rendezvous_points[resources[1]].setReserved(True)

  def getTotalScore(self, optimizer):
    time_a2p, time_p2r, time_r2d, time_r2h = self.getEstimatedTimes4AllEdgesInMinutes()
    remaining_time_all_patients = self.getRemainingTime4AllPatients()

    best_routes = optimizer.getCandidateRoutes(time_a2p, time_p2r, time_r2d, time_r2h, remaining_time_all_patients)

    total_score = 0
    if best_routes:
      #print('best_routes',best_routes)
      for route in best_routes:
        #print('route',route)
        if route[1][0] == -1 or route[1][1] == -1 or route[1][2] == -1 or route[1][3] == -1:
          total_score = None
          break      
        #print('route[4]',route[4])
        total_score += route[3]
        #print(route)
      #print('Total score:',total_score)    
    return total_score

  def addPatients(self, num_of_patients):
    self.num_of_patients += num_of_patients
    for _ in range(num_of_patients):
      self.patients.append(Patient(self.width, self.height))

