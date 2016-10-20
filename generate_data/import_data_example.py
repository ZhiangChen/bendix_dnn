#!/usr/bin/env python
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from visualize_data import Visualizer
import os
import cv2

wd = os.getcwd() # get current work directory
file_name = wd + '/front_dist_data'

with open(file_name,'rb') as f:
	save = pickle.load(f) # load file to pickle
	pos_data = save['pos_data'] # read 'pos_data'
	neg_data = save['neg_data'] # read 'neg_data'
	del save # delete pickle

cm = raw_input('Enter yes to save images\nor Press Enter to continue\n')
if cm == 'yes':
	nm = pos_data.shape[0]
	length = pos_data.shape[1]
	pos_names = ['pos_data_'+str(name)+'.png' for name in range(nm)]
	neg_names = ['neg_data_'+str(name)+'.png' for name in range(nm)]
	t = [0.04*i for i in range(length)]
	for i in range(nm):
		plt.plot(t, pos_data[i,:])
		plt.savefig(pos_names[i], bbox_inches='tight')
		plt.clf()
		plt.plot(t, neg_data[i,:])
		plt.savefig(neg_names[i], bbox_inches='tight')
		plt.clf()

cm = raw_input('Enter 1 to view one pos_data\nEnter 2 to view one neg_data\nor Press Enter to quit\n')
vs = Visualizer()
while cm != '':
	if cm == '1':
		print('Close the figure to continue')
		vs.feed(pos_data,0.04)
		vs.rshow()
	elif cm == '2':
		print('Close the figure to continue')
		vs.feed(neg_data,0.04)
		vs.rshow()
	cm = raw_input('Enter 1 to view one pos_data\n Enter 2 to view one neg_data\n or Press Enter to quit\n')



