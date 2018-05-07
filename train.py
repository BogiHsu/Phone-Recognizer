import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from model import phone_recognizer
from loader import get_mfcc
tf.set_random_seed(0)
np.random.seed(0)

# load timit2mfcc data
print('reading data')
x_train, y_train, _, _, phone_dict = get_mfcc()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.03, random_state = 0)
'''
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:100]
y_test = y_test[:100]
'''
y_train = np.array([np.eye(len(phone_dict))[y] for y in y_train])
y_test = np.array([np.eye(len(phone_dict))[y] for y in y_test])

# init parameters
print('set up parameters')
mfcc_dim = 39
phone_num = len(phone_dict)
layer_num = 1
layer_dim = [512]*layer_num
train_size = x_train.shape[0]
test_size = x_test.shape[0]


epochs = 100
v_period = 5
v_size = (test_size)
save_period = 5
max_keep = 10
batch_size = 1
lr = 0.001
dr = 0.5

x = tf.placeholder(tf.float32, [batch_size, None, mfcc_dim])
y = tf.placeholder(tf.float32, [batch_size, None, phone_num])

# build model
print('building model')
res = phone_recognizer(x, phone_num, batch_size, layer_num, layer_dim, dr)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = res, labels = y))
train_op = tf.train.RMSPropOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(res, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# training step
print('start training')
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep = max_keep)
his = open('./history', 'w')
with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step <= epochs:
		count = 0
		t_acc = 0
		t_loss = 0
		for c in range(len(x_train)):
			count += 1
			batch_xs = np.array([x_train[c]])
			batch_ys = np.array([y_train[c]])
			_, tmpa, tmpc = sess.run([train_op, accuracy, cost], feed_dict = {x:batch_xs, y:batch_ys})
			#sess.run([train_op], feed_dict = {x:batch_xs, y:batch_ys})
			t_acc += tmpa
			t_loss += tmpc
			print('\rEpoch: ', '%3d/%3d'%(step, epochs), ' | Train: ', sep = '', end = '')
			print('%5.1f'%(100*(count/train_size)), '% ', sep = '', end = '')
			print(' acc: %5.3f'%(t_acc/count), ' loss: %5.3f'%(t_loss/count), end = '')
	
		if (step%v_period == 0 or step%save_period == 0):
			acc = 0
			loss = 0
			for c in range(v_size):
				batch_xs = np.array([x_test[c]])
				batch_ys = np.array([y_test[c]])
				acc += sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys})
				loss += sess.run(cost, feed_dict = {x:batch_xs, y:batch_ys})
			if step%v_period == 0:
				his.write('step:'+str(step)+' acc: %5.3f'%(acc/v_size)+' loss: %.3f\n'%(loss/v_size))
				his.flush()
				print(' | Val:', 'acc: %5.3f'%(acc/v_size), ' loss: %.3f'%(loss/v_size), end = '')
			if step%save_period == 0:
				save_path = saver.save(sess, 'models/model-'+str(step).zfill(3)+'-'+str(round((acc/v_size),2))+'.ckpt')
				print('\nsaving model to %s'%save_path, end = '')
		print('')
		step += 1
his.close()
