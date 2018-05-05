import tensorflow as tf
import numpy as np
from model import phone_recognizer
from loader import get_mfcc

# load timit2mfcc data
print('reading data')
x_train, y_train, x_test, y_test, phone_dict = get_mfcc()
y_train = np.array([np.eye(len(phone_dict))[y] for y in y_train])
y_test = np.array([np.eye(len(phone_dict))[y] for y in y_test])

# init parameters
print('set up parameters')
tf.set_random_seed(0)
np.random.seed(0)
mfcc_dim = 39
phone_num = len(phone_dict)
layer_dim = [512, 512, 512]
layer_num = 3

epochs = 200
v_period = 2
save_period = 10
batch_size = 1
lr = 0.001
dr = 0.5

x = tf.placeholder(tf.float32, [batch_size, None, mfcc_dim])
y = tf.placeholder(tf.float32, [batch_size, None, phone_num])

# build model
print('building model')
res = phone_recognizer(x, phone_num, batch_size, layer_num, layer_dim, dr)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = res, labels = y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# training step
print('start training')
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step <= epochs:
		start = np.random.randint(x_train.shape[0])
		count = 1
		for c in range(start, start-x_train.shape[0], -1):
			print('\r', step%v_period, '/', v_period, sep = '', end = ' \t')
			print(round(100*(count/x_train.shape[0]), 1), '\b%', end = '')
			batch_xs = np.array([x_train[c]])
			batch_ys = np.array([y_train[c]])
			sess.run([train_op], feed_dict = {x:batch_xs, y:batch_ys})
			count += 1
		
		if step%v_period == 0:
			acc = 0
			loss = 0
			for c in range(x_test.shape[0]):
				batch_xs = np.array([x_test[c]])
				batch_ys = np.array([y_test[c]])
				acc += sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys})
				loss += sess.run(cost, feed_dict = {x:batch_xs, y:batch_ys})
			print('step: ', step, '\t\tacc: ', acc/x_test.shape[0], '\t\tloss: ', loss/x_test.shape[0], sep = '')

		if step%save_period == 0:
			print('saving')
		step += 1