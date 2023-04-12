# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib import animation, rc
class Animation:
  def __init__(self, greedy, ip, qa, frame=400):
    self.greedy = greedy
    self.ip = ip
    self.qa = qa
    self.fig = plt.figure(figsize=(33, 10))

    grid = plt.GridSpec(1, 3, wspace=0.4, hspace=0.3)
    self.ax1 = plt.subplot(grid[0, 0])
    self.ax2 = plt.subplot(grid[0, 1])
    self.ax3 = plt.subplot(grid[0, 2])
    self.frame = frame

  def initPlot(self):
    return self.greedy.initPlot(self.ax1,'Greedy') + self.ip.initPlot(self.ax2,'Integer Programming') + self.qa.initPlot(self.ax3,'Quantum annealing')

  def update(self, i):
    return self.greedy.update(i) + self.ip.update(i) + self.qa.update(i) 

  def animate(self):
    anim = animation.FuncAnimation(self.fig, self.update, init_func=self.initPlot, frames=self.frame, interval=100, blit=True)
    #anim._func_cid = self.fig.canvas.mpl_connect('resize_event', anim._resize)
    #rc('animation', html='jshtml')
    return anim
