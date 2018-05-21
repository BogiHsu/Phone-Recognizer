import tensorflow as tf
from hyperparams import Hyperparams as hp

def prenet(inputs, is_training):
	outputs = tf.layers.dense(inputs, units=hp.prenet1_size,
						activation=tf.nn.relu)
	outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate,
						training=is_training)
	outputs = tf.layers.dense(outputs, units=hp.prenet2_size,
						activation=tf.nn.relu)
	outputs = tf.layers.dropout(outputs, rate=hp.dropout_rate,
						training=is_training)
	return outputs

def gru(inputs, bidirection, num_units=None):
	if num_units == None:
		num_units = hp.gru_size
	cell = tf.contrib.rnn.GRUCell(num_units,
					activation = tf.tanh,
					kernel_initializer = tf.orthogonal_initializer())
	cell = tf.contrib.rnn.DropoutWrapper(
					cell = cell,
					output_keep_prob = hp.dropout_rate)
	if bidirection:
		cell_bw = tf.contrib.rnn.GRUCell(num_units,
					activation = tf.tanh,
					kernel_initializer = tf.orthogonal_initializer())
		cell_bw = tf.contrib.rnn.DropoutWrapper(
						cell = cell_bw,
						output_keep_prob = hp.dropout_rate)
		outputs, state = tf.nn.bidirectional_dynamic_rnn(
						cell, cell_bw, inputs,
						dtype=tf.float32
					)
		return tf.concat(outputs, 2), tf.concat(state, 1)
	else:
		outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
		return outputs, state

def bn(inputs, is_training, activation_fn=None):
	inputs_shape = inputs.get_shape()
	inputs_rank = inputs_shape.ndims

	# use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
	# pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
	if inputs_rank in [2, 3, 4]:
		if inputs_rank == 2:
			inputs = tf.expand_dims(inputs, axis=1)
			inputs = tf.expand_dims(inputs, axis=2)
		elif inputs_rank == 3:
			inputs = tf.expand_dims(inputs, axis=1)

		outputs = tf.contrib.layers.batch_norm(
					inputs=inputs,
					center=True, scale=True, updates_collections=None,
					is_training=is_training, fused=True,
				  )

		# restore original shape
		if inputs_rank == 2:
			outputs = tf.squeeze(outputs, axis=[1, 2])
		elif inputs_rank == 3:
			outputs = tf.squeeze(outputs, axis=1)

	else:  # fallback to naive batch norm
		outputs = tf.contrib.layers.batch_norm(
					inputs=inputs,
					center=True, scale=True, updates_collections=None,
					is_training=is_training, fused=False)

	if activation_fn is not None:
		outputs = activation_fn(outputs)

	return outputs

def conv1d(inputs, filters=None, size=1, dilation=1,
		   padding="SAME", use_bias=False, activation_fn=None):
	if padding.lower()=="causal":
		# pre-padding for causality
		pad_len = (size - 1) * dilation  # padding size
		inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
		padding = "valid"

	if filters is None:
		filters = inputs.get_shape().as_list[-1]

	params = {"inputs":inputs, "filters":filters, "kernel_size":size,
			  "dilation_rate":dilation, "padding":padding,
			  "activation":activation_fn, "use_bias":use_bias}

	outputs = tf.layers.conv1d(**params)

	return outputs

def conv1d_banks(inputs, K, is_training):
	'''Applies a series of conv1d separately.
	Args:
	inputs: A 3d tensor with shape of [N, T, C]
	K: An int. The size of conv1d banks. That is,
		The `inputs` are convolved with K filters: 1, 2, ..., K.
	is_training: A boolean. This is passed to an argument of `bn`.
	Returns:
		A 3d tensor with shape of [N, T, K*Hp.conv1d_filter_size///2]. '''

	outputs = conv1d(inputs, hp.conv1d_filter_size//2, 1) # k=1
	for k in range(2, K+1): # k = 2...K
		with tf.variable_scope("num_{}".format(k)):
			output = conv1d(inputs, hp.conv1d_filter_size//2, k)
			outputs = tf.concat((outputs, output), -1)

	outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)

	return outputs # (N, T, Hp.embed_size//2*K)

def highwaynet(inputs, num_units=None):
	'''Highway networks
	 Args:
		inputs: A 3D tensor of shape [N, T, W].
		num_units: An int or `None`. Specifies the number of units in the
		highway layer or uses the input size if `None`.
	Returns:
		A 3D tensor of shape [N, T, W].'''
	if not num_units:
		num_units = inputs.get_shape()[-1]
	H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
	T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
		bias_initializer=tf.constant_initializer(-1.0), name="dense2")
	outputs = H*T + inputs*(1.-T)

	return outputs
