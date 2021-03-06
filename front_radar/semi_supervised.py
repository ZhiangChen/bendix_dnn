#!/usr/bin/env python
'''
Zhiang Chen
Dec,2016
'''


from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import time

'''Load Data'''
wd = os.getcwd()
file_name = wd+'/front_dist_data'
with open(file_name, 'rb') as f:
    save = pickle.load(f)
    pos_data = save['pos_data']
    neg_data = save['neg_data']
    del save

#print('pos_data: ',pos_data.shape)
#print('neg_data: ',neg_data.shape)
pos_nm, pos_dim = pos_data.shape
neg_nm, neg_dim = neg_data.shape
assert pos_dim == neg_dim
data_dim = pos_dim


dataset = np.concatenate((pos_data, neg_data), axis=0)
pos_labels = np.asarray([[1,0]]).repeat(pos_nm,axis=0)
neg_labels = np.asarray([[0,1]]).repeat(neg_nm,axis=0)
labels = np.concatenate((pos_labels, neg_labels), axis=0)

#print(np.amax(dataset))
#print(np.amin(dataset))

'''Randomize Data'''
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:]
    shuffled_labels = labels[permutation,:]
    return shuffled_dataset, shuffled_labels

dataset, labels = randomize(dataset, labels)

'''
for i in range(3):
	index = np.random.randint(dataset.shape[0])
	plt.plot(dataset[index,:])
	print('label',labels[index,:])
	plt.show()
'''

'''Assign Dataset'''

data_nm = dataset.shape[0]
train_dataset = dataset[0:int(0.6*data_nm),:].astype(np.float32)
train_labels = labels[0:int(0.6*data_nm),:].astype(np.float32)
valid_dataset = dataset[int(0.6*data_nm):int(0.8*data_nm),:].astype(np.float32)
valid_labels = labels[int(0.6*data_nm):int(0.8*data_nm),:].astype(np.float32)
test_dataset =  dataset[int(0.8*data_nm):,:].astype(np.float32)
test_labels = labels[int(0.8*data_nm):,:].astype(np.float32)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
'''Define MSE & accuracy'''

def MSE(predictions, labels):
	return np.mean((predictions-labels)**2)


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


'''Build Net'''
batch_size = 20
hidden1_nm = 15
hidden2_nm = 3
hidden3_nm = 10
hidden_nm_r = 3
graph = tf.Graph()

with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,data_dim))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,2))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	e_weights1 = tf.Variable(tf.truncated_normal([20,hidden1_nm], stddev=1.0))
	e_biases1 = tf.Variable(tf.zeros([hidden1_nm]))
	e_weights2 = tf.Variable(tf.truncated_normal([hidden1_nm,hidden2_nm], stddev=1.0))
	e_biases2 = tf.Variable(tf.zeros([hidden2_nm]))

	d_weights1 = tf.Variable(tf.truncated_normal([hidden2_nm,hidden1_nm], stddev=1.0))
	d_biases1 = tf.Variable(tf.zeros([hidden1_nm]))
	d_weights2 = tf.Variable(tf.truncated_normal([hidden1_nm,20], stddev=1.0))
	d_biases2 = tf.Variable(tf.zeros([20]))

	e_weights1_r = tf.Variable(tf.truncated_normal([5,hidden_nm_r], stddev=1.0))
	e_biases1_r = tf.Variable(tf.zeros([hidden_nm_r]))

	d_weights1_r = tf.Variable(tf.truncated_normal([hidden_nm_r,5], stddev=1.0))
	d_biases1_r = tf.Variable(tf.zeros([5]))

	weights1 = tf.Variable(tf.truncated_normal([hidden2_nm*2,hidden3_nm], stddev=1.0))
	biases1 = tf.Variable(tf.zeros([hidden3_nm]))

	weights2 = tf.Variable(tf.truncated_normal([hidden3_nm,2], stddev=1.0))
	biases2 = tf.Variable(tf.zeros([2]))

	global_step = tf.Variable(0)  # count the number of steps taken.
	saver = tf.train.Saver()


	def model(data):
		data_f, data_r = data[:,:20], data[:,20:]
		hidden_in = tf.matmul(data_f, e_weights1) + e_biases1
		hidden_out = tf.nn.sigmoid(hidden_in)
		hidden_in = tf.matmul(hidden_out, e_weights2) + e_biases2
		representation_f = tf.nn.sigmoid(hidden_in)

		hidden_in = tf.matmul(data_r, e_weights1_r) + e_biases1_r
		representation_r = tf.nn.sigmoid(hidden_in)

		representation = tf.concat(1,[representation_f, representation_r])

		hidden_in = tf.matmul(representation, weights1) + biases1
		hidden_out = tf.nn.sigmoid(hidden_in)
		o = tf.matmul(hidden_out, weights2) + biases2
		return o

	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
	learning_rate = tf.train.exponential_decay(0.01, global_step, 100000, 0.8, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

start_time = time.time()
nm_steps = 2000000
with tf.Session(graph=graph) as session:
 	saver.restore(session, "autoencoder.ckpt")
 	print('Initialized')
 	for step in range(nm_steps):
 		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
 		batch_data = train_dataset[offset:(offset + batch_size), :]
 		batch_labels = train_labels[offset:(offset + batch_size), :]
 		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
 		lr,_, l, predictions = session.run([learning_rate,optimizer, loss, train_prediction], feed_dict=feed_dict)
 		if (step % 5000 == 0):
 			print('*'*40)
 			print('learning rate: %f'%lr)
 			print('Minibatch loss at step %d: %f' % (step, l))
 			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
 			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
 	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
 	end_time = time.time()
	duration = (end_time - start_time)/60
	print("Excution time: %0.2fmin" % duration)
'''

 	i_test = 0
 	for i_test in np.random.randint(test_dataset.shape[0],size=5).tolist():
 		label = test_labels[int(i_test),:]
 		print('Ground truth label: (%.1f, %.1f)' % (label.tolist()[0],label.tolist()[1]))
 		prd = tf.nn.softmax(model(test_dataset[i_test,:].reshape(1,25))).eval()
 		print('Predicted label: (%.2f, %.2f)' % (prd.tolist()[0][0],prd.tolist()[0][1]))
 	prd_test = test_prediction.eval()
 	prd_index = np.argmax(prd_test,axis=1).reshape(-1,1)
 	org_index = np.argmax(test_labels,axis=1).reshape(-1,1)
 	index_dict = np.concatenate((prd_index,org_index),axis=1)
	pos_dic = np.asarray([x==y==0 for (x,y) in index_dict])
	neg_dic = np.asarray([x==y==1 for (x,y) in index_dict])
	pos_neg_dic = np.asarray([x==0&y==1 for (x,y) in index_dict])
	neg_pos_dic = np.asarray([x==1&y==0 for (x,y) in index_dict])

	prd_pos_data = test_dataset[pos_dic]
	prd_neg_data = test_dataset[neg_dic]
	pos_neg_data = test_dataset[pos_neg_dic]
	neg_pos_data = test_dataset[neg_pos_dic]

def save_image(data,name):
	nm = data.shape[0]
	for i in range(data.shape[0]):
		plt.plot(data[i,:])
		plt.ylim([-0.5,0.5])
		plt.savefig(name+str(i)+'.png')
		plt.clf()

if pos_neg_data.shape[0]!=0:
	save_image(pos_neg_data,'pos_neg')
#save_image(neg_pos_data,'neg_pos')
#save_image(prd_pos_data,'pos')
#save_image(prd_neg_data,'neg')
'''