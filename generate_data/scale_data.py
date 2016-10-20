#!/usr/bin/env python
'''Scale data and save them'''

from generate_data import Simulator
from visualize_data import Visualizer
import numpy as np
import matplotlib.pyplot as plt

class Scaler:

	def __init__(self):
		pass

	def scale_data(self, data):
		nm = data.shape[0]
		length = data.shape[1]
		self.means = np.mean(data,axis=1)
		self.maxs = np.nanmax(data,axis=1)
		self.mins = np.nanmin(data,axis=1)
		max_dist = np.amax(self.maxs)
		min_dist = np.amin(self.mins)
		scaler = max_dist - min_dist
		
		means_mat = np.repeat(self.means,length).reshape((-1,length))
		shifted_data = data - means_mat
		
		scaled_data = shifted_data/scaler		
		return scaled_data

	def show_range_hist(self):
		ranges = self.maxs - self.mins
		plt.bar(range(ranges.shape[0]),ranges)
		plt.show()

if __name__ == '__main__':
	vs = Visualizer()
	pos_data, neg_data, dt = Simulator().get_data(1)
	sc = Scaler()
	scaled_data = sc.scale_data(neg_data)
	vs.feed(neg_data, dt)
	vs.show(1)
	vs.feed(scaled_data,dt)
	vs.show(1)
	sc.show_range_hist()
