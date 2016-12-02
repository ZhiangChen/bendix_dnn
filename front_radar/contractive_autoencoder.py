#!/usr/bin/env python
'''
Zhiang Chen
Dec, 2016
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
hidden1_nm = 30
hidden2_nm = 6
hidden3_nm = 10
graph = tf.Graph()

with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,data_dim))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	e_weights1 = tf.Variable(tf.truncated_normal([data_dim,hidden1_nm], stddev=1.0))
	e_biases1 = tf.Variable(tf.zeros([hidden1_nm]))
	e_weights2 = tf.Variable(tf.truncated_normal([hidden1_nm,hidden2_nm], stddev=1.0))
	e_biases2 = tf.Variable(tf.zeros([hidden2_nm]))

	d_weights1 = tf.Variable(tf.truncated_normal([hidden2_nm,hidden1_nm], stddev=1.0))
	d_biases1 = tf.Variable(tf.zeros([hidden1_nm]))
	d_weights2 = tf.Variable(tf.truncated_normal([hidden1_nm,data_dim], stddev=1.0))
	d_biases2 = tf.Variable(tf.zeros([data_dim]))

	weights1 = tf.Variable(tf.truncated_normal([hidden2_nm,hidden3_nm], stddev=1.0))
	biases1 = tf.Variable(tf.zeros([hidden3_nm]))

	weights2 = tf.Variable(tf.truncated_normal([hidden3_nm,2], stddev=1.0))
	biases2 = tf.Variable(tf.zeros([2]))

	saver = tf.train.Saver()


	def encoder(data):
		hidden_in = tf.matmul(data, e_weights1) + e_biases1
		hidden_out = tf.nn.sigmoid(hidden_in)
		hidden_in = tf.matmul(hidden_out, e_weights2) + e_biases2
		hidden_out = tf.nn.sigmoid(hidden_in)
	  	return hidden_out

	def decoder(data):
		hidden_in = tf.matmul(data, d_weights1) + d_biases1
		hidden_out = tf.nn.sigmoid(hidden_in)
		hidden_in = tf.matmul(hidden_out, d_weights2) + d_biases2
		hidden_out = hidden_in
		return hidden_out

	def reconstruction(data):
		representation = encoder(data)
		pred_data = decoder(representation)
		return pred_data

	cost = tf.reduce_mean(tf.pow(tf_train_dataset - reconstruction(tf_train_dataset), 2))
	optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)

	valid_prediction = reconstruction(tf_valid_dataset)
	test_prediction = reconstruction(tf_test_dataset)

start_time = time.time()
nm_steps = 10000
with tf.Session(graph=graph) as session:
 	tf.initialize_all_variables().run()
 	print('Initialized')
 	for step in range(nm_steps):
 		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
 		batch_data = train_dataset[offset:(offset + batch_size), :]
 		batch_labels = train_labels[offset:(offset + batch_size), :]
 		feed_dict = {tf_train_dataset : batch_data}
 		_, l = session.run([optimizer, cost], feed_dict=feed_dict)
 		if (step % 500 == 0):
 			print('*'*40)
 			print('Minibatch loss at step %d: %f' % (step, l))
 			print('Validation MSE: %f' % MSE(valid_prediction.eval(), valid_dataset))
 	print('Test MSE: %f' % MSE(test_prediction.eval(), test_dataset))
 	end_time = time.time()
	duration = (end_time - start_time)/60
	print("Excution time: %0.2fmin" % duration)
	save_path = saver.save(session, "autoencoder.ckpt")
	print("Model saved in file: %s" % save_path)

 	i_test = 0
 	for i_test in np.random.randint(test_dataset.shape[0],size=10).tolist():
		plt.plot(test_dataset[i_test,:],color='red')
		prd_test = reconstruction(test_dataset[i_test,:].reshape(-1,25)).eval()[0,:]
		plt.plot(prd_test,color='blue')
		plt.ylim([-0.5,0.5])
		plt.show()