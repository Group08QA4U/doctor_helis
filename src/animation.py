# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import animation, rc
class Animation:
  def __init__(self, classic, qa, frame=800):
    self.classic = classic
    self.qa = qa
    self.fig = plt.figure(figsize=(22, 10))

    grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)
    self.ax1 = plt.subplot(grid[0, 0])
    self.ax2 = plt.subplot(grid[0, 1])
    self.frame = frame

  def initPlot(self):
    return self.classic.initPlot(self.ax1,'Cassic') + self.qa.initPlot(self.ax2,'Quantum annealing')

  def update(self, i):
    return self.classic.update(i) + self.qa.update(i)

  def animate(self):
    anim = animation.FuncAnimation(self.fig, self.update, init_func=self.initPlot, frames=self.frame, interval=100, blit=True)
    rc('animation', html='jshtml')
    return anim
