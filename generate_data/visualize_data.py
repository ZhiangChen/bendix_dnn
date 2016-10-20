#!/usr/bin/env python
'''generate raw dataset'''

import numpy as np
import matplotlib.pyplot as plt
from generate_data import Simulator

class Visualizer:

	def __init__(self):
		pass
	
	def feed(self,data,dt):
		self.data = data
		self.nm = data.shape[1]
		self.t = [dt*i for i in range(self.nm)]
		self.length = data.shape[0]

	def rshow(self):
		index = np.random.randint(self.nm)
		plt.plot(self.t, self.data[index,:])
		plt.show()

	def show(self,index):
		plt.plot(self.t, self.data[index,:])
		plt.show()		

if __name__ == '__main__':
	pos_data, neg_data, dt = Simulator().get_data(2)
	vs = Visualizer()
	vs.feed(neg_data,dt)
	vs.rshow()
