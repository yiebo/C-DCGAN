#!/usr/bin/env python
import sys
import os
import numpy as np
import tensorflow as tf
from tqdm import trange

def weight_variable(shape):
	weight = tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
	return weight

def bias_variable(shape):
	bias = tf.get_variable("bias", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
	return bias

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME'):
	return tf.nn.conv2d(x, W, strides, padding='SAME')

def deconv2d(x, W, shape, strides=[1, 1, 1, 1], padding = 'SAME'):
	deconv = tf.nn.conv2d_transpose(value=x, filter=W, output_shape=shape, strides=strides, padding='SAME')
	return tf.reshape(deconv, shape)

def lrelu(x, alpha=0.2, name="lrelu"):
	return tf.maximum(x, alpha*x)

def batch_norm(x, epsilon=1e-5):
	return tf.layers.batch_normalization(inputs=x, epsilon=epsilon)

def shape(tensor):
	return tensor.get_shape().as_list()

def concat_y(x, y):
	with tf.name_scope("concat_y"):
		yb = tf.tile(tf.reshape(y, [-1, 1, 1, shape(y)[-1]]),[1, tf.shape(x)[1], tf.shape(x)[2], 1])
		xy = tf.concat([x, yb], 3)
	return xy


def discriminator(x, y):
	image = tf.reshape(x, [-1, 32, 32, 3])
	image = concat_y(image, y)

	with tf.variable_scope('conv1'):
		W1 = weight_variable([5, 5, shape(image)[-1], 32])
		B1 = bias_variable([32])
		
		L1 = lrelu(conv2d(image, W1, [1,1,1,1]) + B1)


	with tf.variable_scope('conv2'):
		W2 = weight_variable([5, 5, 32, 64])
		B2 = bias_variable([64])
		
		L2 = batch_norm(conv2d(L1, W2, [1,2,2,1]) + B2)
		L2 = lrelu(L2)

		L2 = tf.reshape(L2, [-1, 16*16*64])
	
	with tf.variable_scope('fc1'):
		W3 = weight_variable([16*16*64, 1024])
		B3 = bias_variable([1024])
		
		L3 = batch_norm(tf.matmul(L2, W3) + B3)
		L3 = lrelu(L3)

	L3 = tf.concat([L3, y], 1)

	with tf.variable_scope('fc2'):
		W4 = weight_variable([shape(L3)[-1], 1])
		B4 = bias_variable([1])
		
		L4 = tf.matmul(L3, W4) + B4
	return L4


def generator(z, y):
	batch_s = tf.shape(z)[0]

	z = tf.concat([z, y], 1)
	
	with tf.variable_scope('fc1'):
		W1 = weight_variable([shape(z)[-1], 4*4*1024])
		B1 = bias_variable([4*4*1024])
		
		L1 = tf.nn.relu(tf.matmul(z, W1)+ B1)
		L1 = tf.reshape(L1, [-1, 4, 4, 1024])

	with tf.variable_scope('deconv1'):
		W2 = weight_variable([5, 5, 512, 1024])
		B2 = bias_variable([512])

		L2 = deconv2d(L1, W2, [batch_s, 8, 8, 512], [1,2,2,1]) + B2
		L2 = tf.nn.relu(batch_norm(L2))


	with tf.variable_scope('deconv2'):
		W3 = weight_variable([5, 5, 256, 512])
		B3 = bias_variable([256])
		
		L3 = deconv2d(L2, W3, [batch_s, 16, 16, 256], [1,2,2,1])+B3
		L3 = tf.nn.relu(batch_norm(L3))


	with tf.variable_scope('deconv3'):
		W4 = weight_variable([5, 5, 128, 256])
		B4 = bias_variable([128])
		
		L4 = deconv2d(L3, W4, [batch_s, 32, 32, 128], [1,2,2,1])+B4
		L4 = tf.nn.relu(batch_norm(L4))

	with tf.variable_scope('deconv4'):
		W5 = weight_variable([5, 5, 3, 128])
		B5 = bias_variable([3])
		
		L5 = tf.nn.sigmoid(deconv2d(L4, W5, [batch_s, 32, 32, 3], [1,1,1,1])+B5)

	return L5



if __name__ == '__main__' :
	test_step = 5
	checkpoint_step = 1000

	learning_rate = 0.00005
	training_epochs = 1000000
	batch_size = 5


	images = np.load("cifar10/images.npy")[:50]/255.0
	labels = np.load("cifar10/labels.npy")[:50]/1.0

	x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	z = tf.placeholder(tf.float32, shape=[None, 100])


	with tf.variable_scope("generator") as G:
		g_out = generator(z, y_)

	with tf.variable_scope("discriminator") as D:
		D_real = discriminator(x, y_)
		D.reuse_variables()
		D_false = discriminator(g_out, y_)

	with tf.name_scope("loss"):
		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=(tf.ones_like(D_real)-0.1)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.zeros_like(D_false)))

		D_loss = D_loss_real + D_loss_fake
		G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.ones_like(D_false)))


	G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
	D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

	with tf.name_scope("optimizer"):
		D_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(D_loss, var_list = D_var)
		G_solver = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(G_loss, var_list = G_var)


	x_images = tf.summary.image('True_images', x, 10)
	G_images = tf.summary.image('Generated_images', g_out, 10)

	with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=6)) as sess:
		init = tf.global_variables_initializer()
		sess.run(init)

		saver = tf.train.Saver(max_to_keep=10)

		directory = "GAN_log"
		if not os.path.exists(directory):
			os.makedirs(directory)
		run_nr=len(os.listdir(directory))
		directory = "{}/{}".format(directory, run_nr)

		directory_model = "{}/model".format(directory)
		if not os.path.exists(directory_model):
			os.makedirs(directory_model)

		writer = tf.summary.FileWriter(directory, sess.graph) # for 0.8
		print(directory)
		
		writer.add_summary(sess.run(x_images, feed_dict={x: np.take(images, [29, 4, 6, 9, 10, 27, 0, 7, 8, 1], 0)}),0)


		for i in trange(training_epochs):
			D_cost_total = 0
			G_cost_total = 0
			count = 0
			for idx in trange(0, images.shape[0], batch_size):
				image_batch, label_batch = images[idx:idx+batch_size], labels[idx:idx+batch_size]
				z_set = np.random.uniform(-1., 1., size=[image_batch.shape[0], 100])
				_, D_cost = sess.run([D_solver, D_loss], feed_dict={x: image_batch, y_: label_batch, z: z_set})
				_, G_cost = sess.run([G_solver, G_loss], feed_dict={x: image_batch, y_: label_batch, z: z_set})
				D_cost_total+=D_cost
				G_cost_total+=G_cost
				count+=1
			D_cost_total/=count
			G_cost_total/=count

			summary_train = tf.Summary(value=[tf.Summary.Value(tag="Discriminator", simple_value=D_cost_total)])
			writer.add_summary(summary_train, i)

			summary_train = tf.Summary(value=[tf.Summary.Value(tag="Generator", simple_value=G_cost_total)])
			writer.add_summary(summary_train, i)


			if i%test_step==0:
				z_set = np.random.uniform(-1., 1., size=[10, 100])
				y_test = np.zeros((10, 10))
				np.fill_diagonal(y_test, 1)
				summary = sess.run(G_images, feed_dict={y_: y_test, z: z_set})
				writer.add_summary(summary, i)
				
			if i % checkpoint_step == 0:
				if not os.path.exists('{}/model/iteration{}'.format(directory, i)):
					os.makedirs('{}/model/iteration{}'.format(directory,i))
				saver.save(sess, '{}/model/iteration{}/model.ckpt'.format(directory, i))

