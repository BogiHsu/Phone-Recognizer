import tensorflow as tf
import numpy as np
from model import phone_recognizer
from loader import get_mfcc

# load timit2mfcc data
print('reading data')
x_train, y_train, x_test, y_test, phone_dict = get_mfcc()
'''
x_train = x_train[:20]
y_train = y_train[:20]
x_test = x_test[:20]
y_test = y_test[:20]
'''
y_train = np.array([np.eye(len(phone_dict))[y] for y in y_train])
y_test = np.array([np.eye(len(phone_dict))[y] for y in y_test])

# init parameters
print('set up parameters')
tf.set_random_seed(0)
np.random.seed(0)
mfcc_dim = 39
phone_num = len(phone_dict)
layer_num = 2
layer_dim = [256]*layer_num
train_size = x_train.shape[0]
test_size = x_test.shape[0]


epochs = 400
v_period = 10
v_size = (test_size)//10
save_period = 20
max_keep = 15
batch_size = 1
lr = 0.01
dr = 0.5

x = tf.placeholder(tf.float32, [batch_size, None, mfcc_dim])
y = tf.placeholder(tf.float32, [batch_size, None, phone_num])

# build model
print('building model')
res = phone_recognizer(x, phone_num, batch_size, layer_num, layer_dim, dr)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = res, labels = y))
train_op = tf.train.RMSPropOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(res, 1), tf.argmax(y, 1))
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
		start = np.random.randint(train_size)
		count = 0
		for c in range(start, start-train_size, -1):
			count += 1
			print('\r', 1+(step-1)%v_period, '/', v_period, sep = '', end = '  ')
			print('%5.1f'%(100*(count/train_size)), '%', ' '*10, sep = '', end = '')
			batch_xs = np.array([x_train[c]])
			batch_ys = np.array([y_train[c]])
			sess.run([train_op], feed_dict = {x:batch_xs, y:batch_ys})
	
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
				print('step:', step, ' acc: %5.3f'%(acc/v_size), ' loss: %.3f'%(loss/v_size), ''*10)
			if step%save_period == 0:
				save_path = saver.save(sess, 'models/model-'+str(step).zfill(3)+'-'+str(round((acc/v_size),2))+'.ckpt')
			print('saving model to %s'%save_path)
		step += 1
his.close()
