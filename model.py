import numpy as np
import hyperparams as hp
import tensorflow as tf
from modules import *
from tensorflow.contrib import rnn
np.random.seed(0)
tf.set_random_seed(0)

def phone_recognizer(x, weights, biases, phone_num, batch_size, layer_num, layer_dim, dr = 0.5):
	# BDLSTM
	lstm_fw_cell = rnn.LSTMCell(layer_dim[0]//2, activation = tf.tanh, initializer = tf.orthogonal_initializer(), forget_bias = .5)
	lstm_fw_cell = rnn.DropoutWrapper(cell = lstm_fw_cell, input_keep_prob = 0.9, output_keep_prob = 1-dr)
	lstm_bw_cell = rnn.LSTMCell(layer_dim[1]//2, activation = tf.tanh, initializer = tf.orthogonal_initializer(), forget_bias = .5)
	lstm_bw_cell = rnn.DropoutWrapper(cell = lstm_bw_cell, input_keep_prob = 0.9, output_keep_prob = 1-dr)
	outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
	outputs = tf.concat([outputs[0], outputs[1]], -1)
	
	#outputs.shape = [batch_size, timestep, layer_dim]
	outputs = tf.reshape(outputs, [-1, layer_dim[-1]])
	results = tf.matmul(outputs, weights) + biases
	logits = tf.reshape(results, [batch_size, -1, phone_num])

	return logits

def build_encoder(x, weights, biases, phone_num, batch_size, is_training):
	# x = [batch, seq_len, emb_dim]

	## prenet
	## prenet_out = [batch, seq_len, hp.prenet2_size]
	with tf.variable_scope('encoder_prenet'):
		prenet_out = prenet(x, is_training=is_training)

	## encoder CBHG
	## memory = [batch, seq_len, gru_size]
	## state = [batch, gru_size]
	with tf.variable_scope('encoder_CBHG'):
		## Conv1D banks
		## enc = [batch, seq_len, K*emb_dim/2]
		enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training=is_training)

		## Max pooling
		## enc = [batch, seq_len, K*emb_dim/2]
		enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding="same")
		
		## Conv1D projections
		## enc = [batch, seq_len, emb_dim/2]
		with tf.variable_scope('conv1d_1'):
			enc = conv1d(enc, filters=hp.embed_size//2, size=3)
			enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu)
		with tf.variable_scope('conv1d_2'):
			enc = conv1d(enc, filters=hp.embed_size//2, size=3)
			enc = bn(enc, is_training=is_training, activation_fn=None)
		
		## Residual connections
		## enc = [batch, seq_len, emb_dim/2]
		enc += prenet_out
		
		## Highway Nets
		for i in range(hp.num_highwaynet_blocks):
			with tf.variable_scope('highwaynet_{}'.format(i)):
				enc = highwaynet(enc, num_units=hp.embed_size//2)

		## Bidirectional GRU
		## enc = [batch, seq_len, emb_dim]
		memory, state = gru(enc, num_units=hp.embed_size//2, bidirection=True)
		
	with tf.variable_scope('output_classifier'):
		outputs = tf.reshape(memory, [-1, hp.embed_size])
		results = tf.matmul(outputs, weights) + biases
		logits = tf.reshape(results, [batch_size, -1, phone_num])

	return logits
