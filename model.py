import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np


def phone_recognizer(x, weights, biases, phone_num, batch_size, layer_num, layer_dim, dr = 0.5):
	'''
	# LSTM
	layers = []
	for i in range(layer_num):
		lstm_cell = rnn.LSTMCell(layer_dim[i], activation = tf.tanh, initializer = tf.orthogonal_initializer(), forget_bias = 1.0)
		dr_cell = rnn.DropoutWrapper(cell = lstm_cell, output_keep_prob = 1-dr)
		layers += [dr_cell]
	mlstm_cell = rnn.MultiRNNCell(layers)
	state = mlstm_cell.zero_state(batch_size, dtype = tf.float32)
	outputs, state = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state = state)
	'''
	# BDLSTM
	lstm_fw_cell = rnn.LSTMCell(layer_dim[0], activation = tf.tanh, initializer = tf.orthogonal_initializer(), forget_bias = .0)
	lstm_fw_cell = rnn.DropoutWrapper(cell = lstm_fw_cell, input_keep_prob = 1-dr, output_keep_prob = 1-dr)
	lstm_bw_cell = rnn.LSTMCell(layer_dim[1], activation = tf.tanh, initializer = tf.orthogonal_initializer(), forget_bias = .0)
	lstm_bw_cell = rnn.DropoutWrapper(cell = lstm_bw_cell, input_keep_prob = 1-dr, output_keep_prob = 1-dr)
	outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
	outputs = tf.concat([outputs[0], outputs[1]], -1)
	
	#outputs.shape = [batch_size, timestep, layer_dim]
	outputs = tf.reshape(outputs, [-1, 2*layer_dim[-1]])
	results = tf.matmul(outputs, weights) + biases
	logits = tf.reshape(results, [batch_size, -1, phone_num])

	return logits
	