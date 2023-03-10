# -*- coding: utf-8 -*-

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

