#!/usr/bin/env python
'''Generate, scale, and store data'''

from generate_data import Simulator
from scale_data import Scaler
from six.moves import cPickle as pickle
import os

pos_data, neg_data, dt = Simulator().get_data(500)
sc = Scaler()
scaled_pos_data = sc.scale_data(pos_data)
scaled_neg_data = sc.scale_data(neg_data)

wd = os.getcwd()
data_file = wd + '/front_dist_data'
with open(data_file,'wb') as f:
	save={
		'pos_data': scaled_pos_data,
		'neg_data': scaled_neg_data
	}
	pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
	f.close()
statinfo = os.stat(data_file)
file_size = float(statinfo.st_size)/1000
print('Data size: %0.1fkB' % file_size)
