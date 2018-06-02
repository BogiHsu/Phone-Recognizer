import numpy as np
import hyperparams as hp
import tensorflow as tf
from modules import *
np.random.seed(0)
tf.set_random_seed(0)

def build_encoder(x, phone_num, is_training = True, sf = False):
	# x = [batch, seq_len, emb_dim]
	weights = tf.Variable(tf.random_normal([hp.embed_size, phone_num]))
	biases = tf.Variable(tf.random_normal([phone_num, ]))
	## prenet
	## prenet_out = [batch, seq_len, hp.prenet2_size]
	with tf.variable_scope('encoder_prenet'):
		prenet_out = prenet(x, is_training = is_training)

	## encoder CBHG
	## memory = [batch, seq_len, gru_size]
	## state = [batch, gru_size]
	with tf.variable_scope('encoder_CBHG'):
		## Conv1D banks
		## enc = [batch, seq_len, K*emb_dim/2]
		enc = conv1d_banks(prenet_out, K=hp.encoder_num_banks, is_training = is_training)

		## Max pooling
		## enc = [batch, seq_len, K*emb_dim/2]
		enc = tf.layers.max_pooling1d(enc, pool_size = 2, strides = 1, padding = "same")
		
		## Conv1D projections
		## enc = [batch, seq_len, emb_dim/2]
		with tf.variable_scope('conv1d_1'):
			enc = conv1d(enc, filters = hp.embed_size//2, size = 3)
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
		logits = tf.reshape(results, [hp.batch_size, -1, phone_num])
	
	if sf:
		return tf.nn.softmax(logits)
	return logits
