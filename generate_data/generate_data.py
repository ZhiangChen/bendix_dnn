#!/usr/bin/env python
'''generate raw dataset'''

from oct2py import octave
import os
import numpy as np

class Simulator:

	def __init__(self):
		wd = os.getcwd() # get current work directory in which m-files reside
		octave.addpath(wd) # add current work directory to octave work path
		self.display_figures = 0 # do not display any figures
		pass
	
	def get_data(self,snippets_num):
		
		pos_data = list() # initialize positive data, negative data
		neg_data = list()

		for i in range(snippets_num):
			sim1 = octave.sig_matching_data(1,self.display_figures)
		 	sim2 = octave.sig_matching_data(2,self.display_figures)
			sim3 = octave.sig_matching_data(3,self.display_figures)
			sim4 = octave.sig_matching_data(4,self.display_figures) 
			snippet = octave.match_sigs_create_training_snippets(self.display_figures)
			pos_data.append(snippet['good_example'])
			neg_data.append(snippet['bad_example'])

		self.pos_data = np.asarray(pos_data).reshape((-1,125))
		self.neg_data = np.asarray(neg_data).reshape((-1,125))
		self.dt = sim1['dt']
		return self.pos_data, self.neg_data, self.dt


if __name__ == '__main__':
	pos_data, neg_data, dt = Simulator().get_data(2)
	print(pos_data.shape)
 	print(dt)
	


