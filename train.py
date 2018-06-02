import numpy as np
import tensorflow as tf
from model import *
from loader import get_mfcc
from hyperparams import Hyperparams as hp
np.random.seed(0)
tf.set_random_seed(0)

# load timit2mfcc data
print('reading data')
x_train, x_test, y_train, y_test, mask_train, mask_test, phone_dict, phone_num, max_length = get_mfcc()

# init parameters
print('set up parameters')
train_size = x_train.shape[0]
test_size = x_test.shape[0]
x = tf.placeholder(tf.float32, [hp.batch_size, max_length, hp.mfcc_dim])
y = tf.placeholder(tf.float32, [hp.batch_size, max_length, phone_num])
mask = tf.placeholder(tf.float32, [hp.batch_size, max_length, phone_num])

# build model
print('building model')
res = build_encoder(x, phone_num)
mask_res = tf.multiply(res, mask)
tv = tf.trainable_variables()
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = mask_res, labels = y))
train_op = tf.train.RMSPropOptimizer(hp.lr).minimize(cost)
acc_mask = tf.divide(tf.reduce_sum(mask, axis = -1), tf.constant(phone_num, dtype = 'float32'))
predict = tf.argmax(res, 2)
correct_pred = tf.multiply(tf.cast(tf.equal(predict, tf.argmax(y, 2)), tf.float32), acc_mask)
accuracy = tf.divide(tf.reduce_sum(correct_pred), tf.reduce_sum(acc_mask))

# training step
print('start training')
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver(max_to_keep = hp.max_keep)
his = open('./history-CBHG', 'w')
with tf.Session() as sess:
	sess.run(init[0])
	sess.run(init[1])
	step = 1
	while step <= hp.epochs:
		count = 0
		t_acc = 0
		t_loss = 0
		for c in range(0, len(x_train), hp.batch_size):
			count += 1
			choose = np.random.randint(0, len(x_train), hp.batch_size)
			noise = np.random.normal(1, 0.05, (hp.batch_size, max_length, hp.mfcc_dim))
			batch_xs = np.clip(x_train[choose]*noise, -1, 1)
			batch_ys = y_train[choose]
			batch_masks = mask_train[choose]
			
			_, tmpa, tmpc = sess.run([train_op, accuracy, cost], feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
			t_acc += tmpa
			t_loss += tmpc
			print('\rEpoch: ', '%3d/%3d'%(step, hp.epochs), ' | Train: ', sep = '', end = '')
			print('%3.0f'%(100*(c/train_size)), '% ', sep = '', end = '')
			print(' acc: %5.3f'%(t_acc/count), ' loss: %5.3f'%(t_loss/count), end = '')
		
		if (step%hp.v_period == 0 or step%hp.save_period == 0):
			acc = 0
			loss = 0
			count = 0
			for c in range(0, test_size, hp.batch_size):
				count += 1
				choose = np.random.randint(0, len(x_test), hp.batch_size)
				batch_xs = x_test[choose]
				batch_ys = y_test[choose]
				batch_masks = mask_test[choose]
				acc += sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
				loss += sess.run(cost, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
			if step%hp.v_period == 0:
				his.write('step:'+str(step)+' acc: %5.3f'%(acc/count)+' loss: %.3f\n'%(loss/count))
				his.flush()
				print(' | Val:', 'acc: %5.3f'%(acc/count), ' loss: %.3f'%(loss/count), end = '')
			if step%hp.save_period == 0:
				save_path = saver.save(sess, 'models/model-'+str(step).zfill(3)+'-'+str(round(acc/count, 2))+'.ckpt')
				print('\nsaving model to %s'%save_path, end = '')
		print('')
		step += 1
his.close()
