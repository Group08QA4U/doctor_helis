# -*- coding: utf-8 -*-

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
      #self.remaining_time_to_first_aid = 10         
      self.remaining_time_to_doctor_treatment = max(np.random.normal(loc=20, scale=10.0, size=None),1.0)
      #self.remaining_time_to_basehospital_treatment = 40
    elif self.triage == 2:
      #self.remaining_time_to_first_aid = 20
      self.remaining_time_to_doctor_treatment = max(np.random.normal(loc=40, scale=10.0, size=None),1.0)
      #self.remaining_time_to_basehospital_treatment = 80
      
    else:
      print('ERROR: triage',self.triage,'is not suported')

  def update(self, time):
    self.elapsed_time += time

    #self.remaining_time_to_first_aid -= time
    self.remaining_time_to_doctor_treatment -= time
    #self.remaining_time_to_basehospital_treatment -= time


  def getPosition(self):
    return self.x, self.y

  def setPosition(self,x,y):
    self.x = x
    self.y = y
    return 

  def getRemainingTime(self):
    # 応急処置までの時間(分)、場治療までの時間(分)、病院治療までの時間(分)
    #return self.remaining_time_to_first_aid, self.remaining_time_to_doctor_treatment, self.remaining_time_to_basehospital_treatment
    return self.remaining_time_to_doctor_treatment

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

## 施設ベースクラス
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
