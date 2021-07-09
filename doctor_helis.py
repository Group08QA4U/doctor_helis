import matplotlib.pyplot as plt
import numpy as np

class Plotable:
  def __init__(self):
    self.color = 'black'
    self.marker = '.'
    #self.label = 'object'
    return 

  def getColor(self):
    return self.color

  def getMarker(self):
    return self.marker

  def getLabel(self):
    return self.label




# 要救助者クラス
class Patient(Plotable):
  def __init__(self, x, y):
    super().__init__()
    self.color = 'black'
    self.marker = 'x'    
    self.label = 'patient'

    self.elapsed_time = 0
    self.x = np.random.randint(0, x)
    self.y = np.random.randint(0, y)
    #self.vehicles = []
    #self.clearing_vehicles = []
    # Triage (JTAS)
    # 1: Critical(Priority 1) Urgent 
    # 2: Serious (Priority 2) Can dalay up to 10 mins      
    # 3: Guarded (Priority 3) Can dalay up to 30 mins       
    # 4: Stable  (Priority 4) Can dalay up to 60 mins
    # 5: Dead                 No care needed

    #今回はトリアージは1,2のみを使用する
    self.triage = np.random.randint(1,3)

    # 救命のための応急処置までの時間(分)　remaining_time_to_first_aid
    # 救命のための現場治療までの時間(分)　remaining_time_to_doctor_treatment
    # 救命のための病院治療までの時間(分)　remaining_time_to_basehospital_treatment
    
    # トリアージに応じてランダムにしても良い。TODO
    if self.triage == 1:
      self.remaining_time_to_first_aid = 10         
      self.remaining_time_to_doctor_treatment = 20  
      self.remaining_time_to_basehospital_treatment = 40
    elif self.triage == 2:
      self.remaining_time_to_first_aid = 20
      self.remaining_time_to_doctor_treatment = 40
      self.remaining_time_to_basehospital_treatment = 80
      
    else:
      print('ERROR: triage',self.triage,'is not suported')

  def update(self, time):
    self.elapsed_time += time

    self.remaining_time_to_first_aid -= time
    self.remaining_time_to_doctor_treatment -= time
    self.remaining_time_to_basehospital_treatment -= time


  def getPosition(self):
    return self.x, self.y

  def setPosition(self,x,y):
    self.x = x
    self.y = y
    return 

  def getRemainingTime(self):
    # 応急処置までの時間(分)、場治療までの時間(分)、病院治療までの時間(分)
    return self.remaining_time_to_first_aid, self.remaining_time_to_doctor_treatment, self.remaining_time_to_basehospital_treatment

  def setVehicle(self, vehicle):
    return

  def clearVehicle(self, vehicle):
    return

  def canGoNextTarget(self):
    return True

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()    

  def hasPatient(self):
    return False  


# 施設ベースクラス
class Facility(Plotable):
  def __init__(self, x, y):
    super().__init__()
    self.x = x
    self.y = y
    self.vehicles = []
    self.clearing_vehicles = []
    self.patient = None 

  def getPosition(self):
    return self.x, self.y

  def update(self, time):
    for vehicle in self.clearing_vehicles:
      self.vehicles.remove(vehicle)    
    self.clearing_vehicles = []
    return

  def setVehicle(self, vehicle):
    if vehicle not in self.vehicles:
      self.vehicles.append(vehicle)

  def clearVehicle(self, vehicle):
    if vehicle not in self.clearing_vehicles:
      self.clearing_vehicles.append(vehicle)

  def canGoNextTarget(self):
    return True

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()

  def getOn(self, patient):
    self.patient = patient
    return

  def getOff(self):
    patient = self.patient
    self.patient = None
    return patient

  def hasPatient(self):
    return self.patient != None

  def getPosition(self):
    return self.x, self.y

# 基地病院クラス
class BaseHospital(Facility):
  def __init__(self, x, y):
    super().__init__(x, y)
    self.color = 'orange'
    self.marker = '*'   
    self.label = 'base hospital'   

  def getPosition(self):
    return self.x, self.y

  def update(self, time):
    return super().update(time)

  def setVehicle(self, vehicle):
    super().setVehicle(vehicle)

  def clearVehicle(self, vehicle):
    super().clearVehicle(vehicle)

  def canGoNextTarget(self):
    return super().canGoNextTarget()

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()        

  def getOn(self, patient):
    super().getOn(patient)

  def getOff(self):
    return super().getOff()

  def hasPatient(self):
    return super().hasPatient()


# 消防署クラス
class FireDepartment(Facility):
  def __init__(self,  x, y):
    super().__init__(x, y)
    self.color = 'yellow'
    self.marker = 's'   
    self.label = 'fire department'   

  def getPosition(self):
    return self.x, self.y

  def update(self, time):
    return super().update(time)

  def setVehicle(self, vehicle):
    super().setVehicle(vehicle)

  def clearVehicle(self, vehicle):
    super().clearVehicle(vehicle)

  def canGoNextTarget(self):
    return super().canGoNextTarget()

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()        

  def hasPatient(self):
    return super().hasPatient()

# ランデブーポイント
class RendezvousPoint(Facility):
  def __init__(self, x, y):
    super().__init__(x, y)
    self.color = 'gray'
    self.marker = 'H'   
    self.label = 'rendezvous point'   

    self.is_reserverd = False

  def getPosition(self):
    return super().getPosition()

  def update(self, time):
    return super().update(time)

  def setReserved(self, is_reserverd):
    self.is_reserverd = is_reserverd
    
  def setVehicle(self, vehicle):
    super().setVehicle(vehicle)

  def clearVehicle(self, vehicle):
    super().clearVehicle(vehicle)

  def canGoNextTarget(self):
    #return len(self.vehicles) >= 2
    return (len(self.vehicles) >= 2)

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()        

  def getOn(self, patient):
    super().getOn(patient)

  def getOff(self):
    return super().getOff()

  def hasPatient(self):
    return super().hasPatient()    

  #def setPosition(self):
  #  return 

# 自衛隊基地クラス
class SdfBase(Facility):
  def __init__(self, x, y):
    super().__init__(x, y)
    self.color = 'm'
    self.marker = 's'   
    self.label = 'sdf base'   

  def getPosition(self):
    return self.x, self.y

  def update(self, time):
    return super().update(time)

  def setVehicle(self, vehicle):
    super().setVehicle(vehicle)

  def clearVehicle(self, vehicle):
    super().clearVehicle(vehicle)

  def canGoNextTarget(self):
    return super().canGoNextTarget()

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()    

  def hasPatient(self):
    return super().hasPatient()


# 乗り物ベースクラス
class Vehicle(Plotable):
  def __init__(self, x, y):
    super().__init__()
    self.target_x = self.init_x = self.x = x
    self.target_y = self.init_y = self.y = y
    self.patient = None 
    self.targets = []
    
  def update(self, t):
    # 速度計算
    if self.target_x != self.x:
      velocity_x = (self.velocity * (self.target_x - self.x))/ np.sqrt( (self.target_x - self.x) **2 + (self.target_y - self.y) **2 )
      if (self.target_x - self.x)**2 <= (t * (velocity_x*1000/60))**2:
        self.x = self.target_x
      else:
        self.x += (t * (velocity_x*1000/60))
    if self.target_y != self.y:
      velocity_y = (self.velocity * (self.target_y - self.y))/ np.sqrt( (self.target_x - self.x) **2 + (self.target_y - self.y) **2 )
      if (self.target_y - self.y)**2 <= (t * (velocity_y*1000/60))**2:
        self.y = self.target_y
      else:
        self.y += (t * (velocity_y*1000/60)) 

    # 搬送中の要救助者
    if self.patient != None:
      self.patient.setPosition(self.x, self.y)
    
    # 目的地に到着
    if self.x == self.target_x and self.y == self.target_y and self.x != self.init_x and self.y != self.init_y:

      if self.getLabel() == 'ambulance' and self.targets[0].getLabel() == 'patient':
        if self.hasPatient() != True:
          self.getOn(self.targets[0])

      if self.getLabel() == 'ambulance' and self.targets[0].getLabel() == 'rendezvous point':
        if self.hasPatient() == True :
          self.targets[0].getOn(self.getOff())

      if self.getLabel() == 'doctor heli' and self.targets[0].getLabel() == 'rendezvous point':
        if self.targets[0].hasPatient() != True:
          return 
        if self.hasPatient() != True:
          self.getOn(self.targets[0].getOff())


      if self.getLabel() == 'sdf heli' and self.targets[0].getLabel() == 'patient':
        if self.hasPatient() != True:
          self.getOn(self.targets[0])

      if self.targets[0].getLabel() == 'base hospital':
        if self.hasPatient() == True :
          self.targets[0].getOn(self.getOff())

      self.target_x, self.target_y = self.nextTarget()   


    return 
  def getPosition(self):
    return self.x, self.y
  def getVelocity(self):
    return self.velocity
    
  #def setTarget(self, x, y):
  #  #self.arrived = False
  #  self.target_x = x
  #  self.target_y = y
    

  def setTargets(self, targets):
    self.targets = []
    for target in targets:
      self.targets.append(target)
    self.target_x, self.target_y = self.targets[0].getPosition()

  def nextTarget(self):
    if len(self.targets) == 0:
      print('xxxx')
      return self.init_x, self.init_y

    self.targets.remove(self.targets[0])

    return self.targets[0].getPosition()

  def getTargetPos(self):
    return self.target_x, self.target_y

  def isBusy(self):
    return self.target_x != self.x or self.target_y != self.y

  def plot(self,ax):
    return

  def getOn(self, patient):
    self.patient = patient
    return

  def getOff(self):
    patient = self.patient
    self.patient = None
    return patient

  def setTargetInitPos(self):
    self.target_x = self.init_x
    self.target_y = self.init_y

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()    

  def hasPatient(self):
    return self.patient != None


#ドクターヘリ
class DrHeli(Vehicle):
  def __init__(self, x, y):
    super().__init__(x, y)
    self.color = 'blue'
    self.marker = '1'       
    self.label = 'doctor heli' 
    self.velocity = 200
    return
  def update(self, t):
    return super().update(t)

  def getPosition(self):
    return super().getPosition()    

  def getVelocity(self):
    return super().getVelocity()    

  def isBusy(self, x, y):
    return super().isBusy()

  def setTargetInitPos(self):
    super().setTargetInitPos()

  def getOn(self, patient):
    super().getOn(patient)

  def getOff(self):
    return super().getOff()

  def setTargets(self, objs):
    super().setTargets(objs)

  def nextTarget(self):
    return super().nextTarget()

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()    
    
  def hasPatient(self):
    return super().hasPatient()

#自衛隊ヘリ
class SdfHeli(Vehicle):
  def __init__(self, x, y):
    super().__init__(x, y)
    self.color = 'green'
    self.marker = '2'
    self.label = 'sdf heli'            
    self.velocity = 200

  def update(self, t):
    return super().update(t)

  def getPosition(self):
    return super().getPosition()    

  def getVelocity(self):
    return super().getVelocity()    

  def getTarget(self):
    return super().getTarget()

  def isBusy(self, x, y):
    return super().isBusy()

  def getOn(self, patient):
    super().getOn(patient)

  def getOff(self):
    return super().getOff()

  def setTargets(self, objs):
    super().setTargets(objs)

  def nextTarget(self):
    return super().nextTarget()

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()    
    
  def hasPatient(self):
    return super().hasPatient()

#救急車
class Ambulance(Vehicle):
  def __init__(self, x, y):
    super().__init__(x, y)
    self.color = 'red'
    self.marker = '+'       
    self.label =  'ambulance'
    self.velocity = 60

  def update(self, t):
    return super().update(t)

  def getPosition(self):
    return super().getPosition()    

  def getVelocity(self):
    return super().getVelocity()      
      
  def getTarget(self):
    return super().getTarget()

  def isBusy(self, x, y):
    return super().isBusy()

  def setTargetInitPos(self):
    super().setTargetInitPos()

  def getOn(self, patient):
    super().getOn(patient)

  def getOff(self):
    return super().getOff()    

  def setTargets(self, objs):
    super().setTargets(objs)

  def nextTarget(self):
    return super().nextTarget()    

  def getColor(self):
    return super().getColor()

  def getMarker(self):
    return super().getMarker()

  def getLabel(self):
    return super().getLabel()   
    
  def hasPatient(self):
    return super().hasPatient()

import math
from numpy import linalg as LA
from matplotlib import animation, rc
from IPython.display import HTML

class World:
  def __init__(self, width, height, num_of_patients, num_of_rendezvous_points, num_of_sdf_bases, num_of_basehospitals, num_of_fire_departments):

    #仮想地図の解像度（メートル）10km x 10km
    self.width = width
    self.height = height

    self.step = 0.0 # minuts
    self.time = 0

    self.num_of_patients = num_of_patients
    self.num_of_ambulances = self.num_of_fire_departments = num_of_fire_departments
    self.num_of_rendezvous_points = num_of_rendezvous_points
    self.num_of_doctor_helis = self.num_of_basehospitals = num_of_basehospitals
    self.num_of_sdf_helis = self.num_of_sdf_bases = num_of_sdf_bases


    # 要救助者
    self.patients = []
    for _ in range(self.num_of_patients):
      self.patients.append(Patient(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 消防署
    self.fire_departments = []
    for _ in range(self.num_of_fire_departments):
      self.fire_departments.append(FireDepartment(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 救急車 
    self.ambulances = []
    for i in range(self.num_of_ambulances):
      x, y = self.fire_departments[i].getPosition()
      self.ambulances.append(Ambulance(x, y))      

    # ランデブーポイント
    self.rendezvous_points = []
    for _ in range(self.num_of_rendezvous_points):
      self.rendezvous_points.append(RendezvousPoint(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 基地病院
    self.basehospitals = []
    for _ in range(self.num_of_basehospitals):
      self.basehospitals.append(BaseHospital(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # ドクターヘリ
    self.doctor_helis = []
    for i in range(self.num_of_doctor_helis):
      x, y = self.basehospitals[i].getPosition() 
      self.doctor_helis.append(DrHeli(x, y))

    # 自衛隊基地
    self.sdf_bases = []
    for _ in range(self.num_of_sdf_bases):
      self.sdf_bases.append(SdfBase(np.random.randint(0, self.width), np.random.randint(0, self.height)))

    # 自衛隊ヘリ
    self.sdf_helis = []
    for i in range(self.num_of_sdf_helis):
      x, y = self.sdf_bases[i].getPosition() 
      self.sdf_helis.append(SdfHeli(x, y))

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

    # 自衛隊基地
    if len(self.sdf_bases) > 0:
      c = self.sdf_bases[0].getColor()
      m = self.sdf_bases[0].getMarker()
      l = self.sdf_bases[0].getLabel()       
      self.point_sdf_bases = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)   
      offset = []
      for sdf_base in self.sdf_bases:
        x, y = sdf_base.getPosition()
        offset.append([x, y])
      self.charts.append(self.point_sdf_bases)

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

    # 自衛隊ヘリ
    if len(self.sdf_helis) > 0:
      c = self.sdf_helis[0].getColor()
      m = self.sdf_helis[0].getMarker()
      l = self.sdf_helis[0].getLabel()     
      self.route_sdf_helis = []
      self.point_sdf_helis = self.ax.scatter([], [], c=c, marker=m, label=l, s=200)   
      self.charts.append(self.point_sdf_helis)
      for sdf_heli in self.sdf_helis:
        line, = self.ax.plot([], [], c=c, linestyle="dashed")
        self.route_sdf_helis.append(line)      
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
        first_aid, doctor_treatment, basehospital_treatment = patient.getRemainingTime()
        fc='green'
        if first_aid <= 5 or doctor_treatment <= 5 or basehospital_treatment <= 5:
          fc='yellow'
        if first_aid <= 1 or doctor_treatment <= 1 or basehospital_treatment <= 0:
          fc='red'  
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

    # 自衛隊基地
    if len(self.sdf_bases):
      offset = []
      for sdf_base in self.sdf_bases:
        x, y = sdf_base.getPosition()
        offset.append([x, y])
      self.point_sdf_bases.set_offsets(offset)

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

    # 自衛隊ヘリ
    if len(self.sdf_helis):
      offset = []
      for i, sdf_heli in enumerate(self.sdf_helis):
        x, y = sdf_heli.getPosition()
        offset.append([x, y])
        target_x, target_y = sdf_heli.getTargetPos()
        self.route_sdf_helis[i].set_data([x, target_x], [y, target_y])
      self.point_sdf_helis.set_offsets(offset)

    # 要救助者
    if len(self.patients):
      offset = []
      for i, patient in enumerate(self.patients):
        x, y = patient.getPosition()
        offset.append([x, y])
        first_aid, doctor_treatment, basehospital_treatment = patient.getRemainingTime()
        fc='green'
        if first_aid <= 5 or doctor_treatment <= 5 or basehospital_treatment <= 5:
          fc='yellow'
        if first_aid <= 1 or doctor_treatment <= 1 or basehospital_treatment <= 0:
          fc='red'  
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
    # 自衛隊ヘリ速度（時速）
    if len(self.sdf_helis):
      velocity_sdf_heli = self.sdf_helis[0].getVelocity()

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
    # 自衛隊ヘリ to 要救助者
    self.dist_s2p = np.zeros((self.num_of_patients, self.num_of_sdf_helis))
    self.time_s2p = np.zeros((self.num_of_patients, self.num_of_sdf_helis))
    # 要救助者 to 基地病院
    self.dist_p2h = np.zeros((self.num_of_patients, self.num_of_basehospitals))
    self.time_p2h = np.zeros((self.num_of_patients, self.num_of_basehospitals))
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
      # 自衛隊ヘリから要救助者への全ルートの距離を計算
      if len(self.sdf_helis):      
        for k, sdf_heli in enumerate(self.sdf_helis):
          x2 = sdf_heli.getPosition()[0]
          y2 = sdf_heli.getPosition()[1]
          self.dist_s2p[i][k] = self.distance(x1, y1, x2, y2)
          self.time_s2p[i][k] = (self.distance(x1, y1, x2, y2)*60)/(velocity_sdf_heli * 1000)
      # 要救助者から基地病院への全ルートの距離を計算
      if len(self.basehospitals):        
        for l, basehospital in enumerate(self.basehospitals):
          x2 = basehospital.getPosition()[0]
          y2 = basehospital.getPosition()[1]
          self.dist_p2h[i][l] = self.distance(x1, y1, x2, y2)
          self.time_p2h[i][l] = (self.distance(x1, y1, x2, y2)*60)/(velocity_sdf_heli * 1000)

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
    # time_p2h : 要救助者(自衛隊ヘリ)から基地病院
    # time_r2d ; ランデブーポイントからドクターヘリ
    # time_r2h : ランデブーポイントから基地病院
    return self.time_a2p, self.time_p2r, self.time_s2p, self.time_p2h, self.time_r2d, self.time_r2h

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

    # 自衛隊ヘリ
    for i, sdf_heli in enumerate(self.sdf_helis):
      sdf_heli.update(self.step)

    # 基地病院
    for basehospital in self.basehospitals:
      basehospital.update(self.step)

    
    self.step = 0.1
    self.time += self.step
    return self.plot()

  def despatch(self, optimizer):
    time_a2p, time_p2r, time_s2p, time_p2h,time_r2d, time_r2h = self.getEstimatedTimes4AllEdgesInMinutes()
    remaining_time_all_patients = self.getRemainingTime4AllPatients()
    
    best_routes = optimizer.getCandidateRoutes(time_a2p, time_p2r, time_s2p, time_p2h,time_r2d, time_r2h, remaining_time_all_patients)

    print('[patient#, [ambulance to patient, ambulance to rendezvous_point, rendezvous_point to doctor_heli, doctor_heli to base_hospital, sdf_heli to patient, patient to base hospital], estimate time]')
    for route in best_routes:
      print(route)

    #  p, [a2p,p2r,r2d,d2h,s2p,p2h], estimated_time
    #[[0, [-1, -1, -1, -1, 2, 0], 2.058369153477586], 
    # [1, [-1, -1, -1, -1, 1, 1], 1.544260844635423],
    # [2, [-1, -1, -1, -1, 0, 0], 1.7684121767858059],
    # [3, [3, 2, 0, 1, -1, -1], 8.212439216185327],
    # [4, [2, 0, 1, 0, -1, -1], 6.989918761281399]]
    self.current_routes = best_routes

    for p, resources, _ in self.current_routes:
      if resources[1] != -1 and resources[0] != -1:
        self.ambulances[resources[0]].setTargets([self.patients[p], self.rendezvous_points[resources[1]], self.fire_departments[resources[0]]])

      if resources[2] != -1 and resources[1] != -1:        
        self.doctor_helis[resources[2]].setTargets([self.rendezvous_points[resources[1]], self.basehospitals[resources[2]]])

      if resources[4] != -1 and resources[5] != -1:
        self.sdf_helis[resources[4]].setTargets([self.patients[p], self.basehospitals[resources[5]] ,self.sdf_bases[resources[4]]])

      if resources[1] != -1:
        self.rendezvous_points[resources[1]].setReserved(True)


  def addPatients(self, num_of_patients):
    self.num_of_patients += num_of_patients
    for _ in range(num_of_patients):
      self.patients.append(Patient(self.width, self.height))


class Animation:
  def __init__(self, classic, qa):
    self.classic = classic
    self.qa = qa
    self.fig = plt.figure(figsize=(22, 10))

    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    self.ax1 = plt.subplot(grid[0, 0])
    self.ax2 = plt.subplot(grid[0, 1])


  def initPlot(self):
    return self.classic.initPlot(self.ax1,'Cassic') + self.qa.initPlot(self.ax2,'Quantum annealing')

  def update(self, i):
    return self.classic.update(i) + self.qa.update(i)

  def animate(self):
    anim = animation.FuncAnimation(self.fig, self.update, init_func=self.initPlot, frames=250, interval=100, blit=True)
    rc('animation', html='jshtml')
    return anim


import sys
# 最適化ベースクラス
class Optimizer:
  def __init__(self):
    return

  def getCandidateRoutes(self, time_a2p, time_p2r, time_s2p, time_p2h, time_r2d, time_r2h, remaining_time_all_patients, is_debug=False):

    self.num_of_patients = time_a2p.shape[1]
    self.num_of_ambulances = time_a2p.shape[0]
    self.num_of_rendezvous_points = time_p2r.shape[1]
    self.num_of_doctor_helis = time_r2d.shape[1]
    self.num_of_sdf_helis = time_s2p.shape[1]
    self.num_of_basehospitals = time_r2h.shape[1]

    self.time_a2p = time_a2p
    self.time_p2r = time_p2r
    self.time_s2p = time_s2p
    self.time_p2h = time_p2h
    self.time_r2d = time_r2d
    self.time_r2h = time_r2h

    self.remaining_time_all_patients = remaining_time_all_patients
    
    if is_debug:
      print('num of patients',self.num_of_patients)
      print('num of ambulances',self.num_of_ambulances)
      print('num of rendezvous_points',self.num_of_rendezvous_points)
      print('num of doctor_helis',self.num_of_doctor_helis)
      print('num of sdf_helis',self.num_of_sdf_helis)
      print('num of basehospitals',self.num_of_basehospitals)
          
      print('Estimate time from ambulance to patient (ambulance x patient)\n',self.time_a2p)
      print('Estimate time from patient to rendezvous_point (patient x rendezvous_point)\n',self.time_p2r)
      print('Estimate time from patient to SDF heli (patient x SDF heli)\n',self.time_s2p)
      print('Estimate time from patient to basehospital (patient x basehospital)\n',self.time_p2h)
      print('Estimate time from rendezvous_point to doctor heli (rendezvous_point x doctor heli)\n',self.time_r2d)
      print('Estimate time from rendezvous_point to basehospital (rendezvous_point x basehospital)\n',self.time_r2h)    

      print('remaining time for all patients [(FirstAid, Treatement, Tobasehospital),]\n',remaining_time_all_patients)




class Classic(Optimizer):
  def __init__(self):
    super().__init__()

  def getCandidateRoutes(self, time_a2p, time_p2r, time_s2p, time_p2h, time_r2d, time_r2h, remaining_time_all_patients):
    super().getCandidateRoutes(time_a2p, time_p2r, time_s2p, time_p2h, time_r2d, time_r2h, remaining_time_all_patients)
    
    candidate_routes = np.zeros((self.num_of_patients, self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_sdf_helis * self.num_of_basehospitals ))
    reserved_ambulances = np.zeros(self.num_of_ambulances)
    reserved_rendezvous_points = np.zeros(self.num_of_rendezvous_points)
    reserved_doctor_helis = np.zeros(self.num_of_doctor_helis)
    reserved_sdf_helis = np.zeros(self.num_of_sdf_helis)
    reserved_basehospitals = np.zeros(self.num_of_basehospitals)
    
    best_routes = []
    for i in range(self.num_of_patients):
      min_route1 = min_route2 = sys.maxsize
      a2p = p2r = r2d = d2h = -1      
      s2p = p2h = -1   
      for j in range(self.num_of_ambulances):
        if reserved_ambulances[j] == True:
          continue 
        for k in range(self.num_of_rendezvous_points):
          if reserved_rendezvous_points[k] == True:
            continue           
          for l in range(self.num_of_doctor_helis):
            if reserved_doctor_helis[l] == True:
              continue   
            for m in range(self.num_of_basehospitals):
              route_estimated_time = max( time_a2p[j][i] + time_p2r[i][k] , time_r2d[k][l] ) + time_r2h[k][m]
              #print(i,j,k,l,m,-1,-1,route_estimated_time)
              if min_route1 > route_estimated_time:
                min_route1 = route_estimated_time
                a2p = j
                p2r = k
                r2d = l
                d2h = m

         
      for n in range(self.num_of_sdf_helis):
        if reserved_sdf_helis[n] == True:
          continue         
        for o in range(self.num_of_basehospitals):
          route_estimated_time = time_s2p[i][n] + time_p2h[i][o]
          #print(i,-1,-1,-1,-1,n,o,route_estimated_time)
          if min_route2 > route_estimated_time:
            min_route2 = route_estimated_time
            s2p = n
            p2h = o

      if min_route1 < min_route2:
        reserved_ambulances[a2p] = True
        reserved_rendezvous_points[p2r] = True
        reserved_doctor_helis[r2d] = True
        best_routes.append([i, [a2p, p2r, r2d, d2h, -1, -1], min_route1])

      else:
        reserved_sdf_helis[s2p] = True        
        best_routes.append([i, [-1, -1, -1, -1, s2p, p2h], min_route2] )


    return best_routes    



from dwave.system import LeapHybridSampler, EmbeddingComposite
from openjij import SQASampler

class QA(Optimizer):
  def __init__(self):
    super().__init__()
    #self.token = '***************' 
    #self.endpoint = 'https://cloud.dwavesys.com/sapi/'

  def getCandidateRoutes(self, time_a2p, time_p2r, time_s2p, time_p2h, time_r2d, time_r2h, remaining_time_all_patients):
    super().getCandidateRoutes(time_a2p, time_p2r, time_s2p, time_p2h, time_r2d, time_r2h, remaining_time_all_patients)
    num_route = self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_basehospitals + self.num_of_sdf_helis * self.num_of_basehospitals
    N = self.num_of_patients * num_route
    route_estimated_time = []
    candidate_routes = []
    diff_time = np.zeros(self.num_of_patients*num_route).reshape(self.num_of_patients,num_route)

    #print(remaining_time_all_patients)
    #print(remaining_time_all_patients[0][2])
    
    # 生存時間、搬送時間の計算
    for i in range(self.num_of_patients):
      for j in range(self.num_of_ambulances):
        for k in range(self.num_of_rendezvous_points):    
          for l in range(self.num_of_doctor_helis):
            for m in range(self.num_of_basehospitals):
              n_route =  j*self.num_of_rendezvous_points + k*self.num_of_doctor_helis + l*self.num_of_basehospitals + m
              #route_estimated_time.append( time_a2p[j][i] + time_p2r[i][k] + time_r2d[k][l] + time_r2h[k][m] )
              #print(remaining_time_all_patients[i][2], time_a2p[j][i] + time_p2r[i][k] + time_r2d[k][l] + time_r2h[k][m])
              diff_time[i, n_route] = remaining_time_all_patients[i][2] - (max(time_a2p[j][i] + time_p2r[i][k], time_r2d[k][l]) + time_r2h[k][m])
              candidate_routes.append([i,[j,k,l,m,-1,-1], diff_time[i, n_route]])

      for n in range(self.num_of_sdf_helis):  
        for o in range(self.num_of_basehospitals):
          n_route = n*self.num_of_basehospitals + o + self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_basehospitals
          #route_estimated_time.append( time_s2p[i][n] + time_p2h[i][o] )
          diff_time[i, n_route] = remaining_time_all_patients[i][2] - (time_s2p[i][n] + time_p2h[i][o]) 
          candidate_routes.append([i,[-1,-1,-1,-1,n,o], diff_time[i, n_route]])

    #print(np.array(route_estimated_time).shape)
    #print(route_estimated_time)
    #print(np.array(candidate_routes).shape)
    #print(candidate_routes)
            
    # 経路の選択を反発させるコストの計算
    edge_dict = {}
    
    addr_ambulances = self.num_of_patients
    addr_rendezvous_points = self.num_of_patients*self.num_of_ambulances
    addr_doctor_helis = self.num_of_patients*self.num_of_ambulances*self.num_of_rendezvous_points
    addr_basehospitals = self.num_of_patients*self.num_of_ambulances*self.num_of_rendezvous_points*self.num_of_doctor_helis
    addr_sdf_helis = addr_basehospitals + self.num_of_basehospitals
    addr_basehospitals = addr_basehospitals + self.num_of_basehospitals*self.num_of_sdf_helis
    
    laddr = [addr_ambulances, addr_rendezvous_points, addr_doctor_helis, addr_basehospitals, addr_sdf_helis, addr_basehospitals]
    #print(laddr)
    
    route_add = 0
    for j in range(self.num_of_ambulances):
      e = (j+addr_ambulances)
      if e not in edge_dict.keys():
        edge_dict[e] = route_add
        route_add += 1
        
    for l in range(self.num_of_doctor_helis):
      e = (l+addr_doctor_helis)
      if e not in edge_dict.keys():
        edge_dict[e] = route_add
        route_add += 1

    for n in range(self.num_of_sdf_helis):
      e = (n+addr_sdf_helis)
      if e not in edge_dict.keys():
        edge_dict[e] = route_add
        route_add += 1
    
    #print(route_add)
    #print(edge_dict)

    route_cost = np.zeros(N*len(edge_dict)).reshape(len(edge_dict),N)
    for i in range(self.num_of_patients):
      for k in range(num_route):
        m = k+i*self.num_of_patients
        for l in (0, 2, 4):
          if (candidate_routes[m][1][l] != -1 and candidate_routes[m][1][l+1] != -1):
            #print(m,l,candidate_routes[m][1], candidate_routes[m][1][l]+laddr[l], candidate_routes[m][1][l+1]+laddr[l+1])
            e = edge_dict[(candidate_routes[m][1][l]+laddr[l])]
            #print(laddr[l],laddr[l+1],e)
            route_cost[e,m] = 1
            
    #print(route_cost)
            
    qubo = self.QUBO(diff_time, route_cost)
    
    Nsamples=10
    
    # d-wave hybrid
    #sampler = LeapHybridSampler(solver='hybrid_binary_quadratic_model_version2', token=self.token, endpoint=self.endpoint)
    #results = sampler.sample_qubo(qubo)
    
    # OpenJIJ
    sampler = SQASampler(num_sweeps = 3000)
    results = sampler.sample_qubo(qubo, num_reads=Nsamples)
    
    # 結果加工
    # 0番目のデータを使う（エネルギーが一番小さい値を選ぶ処理が必要かも？）
    results_jij = np.zeros(N)
    results_jij[:] = results.record[0][0]

    print(results)
    return self.postProcessing(results_jij, candidate_routes)

  def postProcessing(self, results, candidate_routes):
    # D-waveの結果を使用しやすいデータに加工する
    
    dw_answer_routes = []
    answer_list = np.where(results==1)[0]
    for m in answer_list:
        print(m)
        dw_answer_routes.append(candidate_routes[m])
    
    return dw_answer_routes


  def QUBO(self, diff_time, route_cost):
    # Make QUBO from time_a2p, time_p2r... and remaining_time_all_patients
    num_route = self.num_of_ambulances * self.num_of_rendezvous_points * self.num_of_doctor_helis * self.num_of_basehospitals + self.num_of_sdf_helis * self.num_of_basehospitals
    N = self.num_of_patients * num_route
    
    # lam1: 要救助に1つ搬送経路を割り当てる制約条件の強さ
    # lam2: 生存時間のコスト項の強さ
    lam1 = 25.0
    lam2 = 0.2
    lam3 = 25.0
    
    #lam1 = 30.0
    #lam2 = 0.0
    
    # QUBO行列生成
    Q1 = np.zeros(N*N).reshape(N,N)
    for i in range(self.num_of_patients):
      for j in range(num_route):
        for l in range(num_route):
          m1 = j + i*num_route
          m2 = l + i*num_route
          Q1[m1,m2] = Q1[m1,m2] + 1.0*lam1
          if m1 == m2:
            Q1[m1,m2] = Q1[m1,m2] - 2.0*lam1 - lam2*diff_time[i, l]
    
    Q2 = np.dot(route_cost.T,route_cost)
    
    Q = Q1 + lam3*Q2
    
    # 辞書形式にする
    qubo = {}
    for m in range(N):
      for n in range(N):
        if Q[m,n] != 0.0:
          qubo[m,n] = Q[m,n]

    return qubo


import copy

# 古典コンピューターで計算
world_classic = World(width = 10000, height = 10000, num_of_patients=9, num_of_rendezvous_points = 10, num_of_sdf_bases = 6, num_of_basehospitals = 3, num_of_fire_departments = 4)
world_qa = copy.deepcopy(world_classic)

classic = Classic()
world_classic.despatch(classic)

# QAで計算
#world_qa  = World(width = 10000, height = 10000, num_of_patients=9, num_of_rendezvous_points = 10, num_of_sdf_bases = 6, num_of_basehospitals = 3, num_of_fire_departments = 4)
qa = QA()
world_qa.despatch(qa)


anim = Animation(world_classic, world_qa).animate()

