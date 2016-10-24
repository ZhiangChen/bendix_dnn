#!/usr/bin/env python

from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

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
'''Define MSE'''

def MSE(predictions, labels):
	return np.mean((predictions-labels)**2)


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])



batch_size = 2000

graph = tf.Graph()

with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,data_dim))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,2))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)

	weights = tf.Variable(tf.truncated_normal([data_dim,2], stddev=0.2))
	biases = tf.Variable(tf.zeros([2]))

	def model(data):
		linear = tf.matmul(data, weights) + biases
		return linear

	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

nm_steps = 100
with tf.Session(graph=graph) as session:
 	tf.initialize_all_variables().run()
 	print('Initialized')
 	for step in range(nm_steps):
 		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
 		batch_data = train_dataset[offset:(offset + batch_size), :]
 		batch_labels = train_labels[offset:(offset + batch_size), :]
 		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
 		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
 		if (step % 10 == 0):
 			print('*'*40)
 			print('Minibatch loss at step %d: %f' % (step, l))
 			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
 			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
 	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
 	i_test = 0
 	for i_test in np.random.randint(test_dataset.shape[0],size=5).tolist():
 		label = test_labels[int(i_test),:]
 		print('Ground truth label: (%.1f, %.1f)' % (label.tolist()[0],label.tolist()[1]))
 		prd = tf.nn.softmax(model(test_dataset[i_test,:].reshape(1,25))).eval()
 		print('Predicted label: (%.2f, %.2f)' % (prd.tolist()[0][0],prd.tolist()[0][1]))
