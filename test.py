import numpy as np
import tensorflow as tf
from model import *
from loader import get_mfcc
from hyperparams import Hyperparams as hp
np.random.seed(0)
tf.set_random_seed(0)

# load mfcc data
print('reading data')
_, x_test, _, y_test, _, mask_test, phone_dict, phone_num, max_length = get_mfcc()

# init parameters
print('set up parameters')
test_size = x_test.shape[0]
x = tf.placeholder(tf.float32, [hp.batch_size, max_length, hp.mfcc_dim])
y = tf.placeholder(tf.float32, [hp.batch_size, max_length, phone_num])
mask = tf.placeholder(tf.float32, [hp.batch_size, max_length, phone_num])

# build model
print('building model')
res = build_encoder(x, phone_num, False)
mask_res = tf.multiply(res, mask)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = mask_res, labels = y))
acc_mask = tf.divide(tf.reduce_sum(mask, axis = -1), tf.constant(phone_num, dtype = 'float32'))
predict = tf.argmax(res, 2)
correct_pred = tf.multiply(tf.cast(tf.equal(predict, tf.argmax(y, 2)), tf.float32), acc_mask)
accuracy = tf.divide(tf.reduce_sum(correct_pred), tf.reduce_sum(acc_mask))

def fast_filter(pre):
	for i in range(1, len(pre)-1):
		if pre[i-1] == pre[i+1] and pre[i] != pre[i-1]:
			pre[i] = pre[i-1]
			i += 1
	return pre

# testing step
print('start testing')
with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint('./models-bound-mel/'))
	acc = 0
	loss = 0
	raw_acc = 0
	filt_acc = 0
	count = 0
	for c in range(0, test_size, hp.batch_size):
		#print('\r%5.1f'%(100*(c/test_size)), '% ', end = '')
		if c+hp.batch_size > test_size:
			batch_xs = x_test[-1*hp.batch_size:]
			batch_ys = y_test[-1*hp.batch_size:]
			batch_masks = mask_test[-1*hp.batch_size:]
		else:
			batch_xs = x_test[c:c+hp.batch_size]
			batch_ys = y_test[c:c+hp.batch_size]
			batch_masks = mask_test[c:c+hp.batch_size]
		pre = sess.run(predict, feed_dict = {x:batch_xs, mask:batch_masks})
		# acc from tf
		acc += sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		loss += sess.run(cost, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		# acc before filter
		batch_ys = np.argmax(batch_ys, 2)
		batch_masks = batch_masks[:, :, 0]
		raw_same = np.equal(pre, batch_ys).astype(float)*batch_masks
		raw_acc += np.sum(raw_same)/np.sum(batch_masks)
		# acc after filter
		pre = np.array([fast_filter(pre[i]) for i in range(len(pre))])
		filt_same = np.equal(pre, batch_ys).astype(float)*batch_masks
		filt_acc += np.sum(filt_same)/np.sum(batch_masks)
		# count batchs
		count += 1
		#choose = 3
		#print(phone_dict[pre[choose][i]], phone_dict[np.argmax(batch_ys, 2)[choose][i]])
	print('\nTF:', 'acc: %5.3f'%(acc/count), ' loss: %.3f'%(loss/count))
	print('BF:', 'acc: %5.3f'%(raw_acc/count))
	print('AF:', 'acc: %5.3f'%(filt_acc/count))
