import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def phone_recognizer(x, phone_num, batch_size, layer_num, layer_dim, dr = 0.5):
	weights = tf.Variable(tf.random_normal([layer_dim[-1], phone_num]))
	biases = tf.Variable(tf.zeros([phone_num, ]))
	layers = []
	for i in range(layer_num):
		lstm_cell = rnn.BasicLSTMCell(layer_dim[i], activation = tf.tanh)
		dr_cell = rnn.DropoutWrapper(cell = lstm_cell, output_keep_prob = 1-dr)
		layers += [dr_cell]
	mlstm_cell = rnn.MultiRNNCell(layers)
	state = mlstm_cell.zero_state(batch_size, dtype = tf.float32)
	outputs, state = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state = state)
	#outputs.shape = [batch_size, timestep, layer_dim]
	outputs = tf.reshape(outputs, [-1, layer_dim[-1]])
	results = tf.matmul(outputs, weights) + biases
	logits = tf.reshape(results, [batch_size, -1, phone_num])

	return logits
	