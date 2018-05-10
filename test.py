import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from model import phone_recognizer
from loader import get_mfcc
tf.set_random_seed(0)
np.random.seed(0)

# load timit2mfcc data
print('reading data')
_, o_y_train, o_x_test, o_y_test, phone_dict = get_mfcc()

# preprocessing
print('preprocessing')
num_samples = o_x_test.shape[0]
max_length = max([len(train) for train in o_y_train]+[len(test) for test in o_y_test])
mfcc_dim = 39
x_test = np.zeros([num_samples, max_length, mfcc_dim])
y_test = np.zeros([num_samples, max_length, len(phone_dict)], dtype = 'int8')
mask_test = np.zeros([num_samples, max_length, len(phone_dict)], dtype = 'int8')
for i in range(num_samples):
	x_test[i, :len(o_x_test[i]), :] = o_x_test[i]/60
	y_test[i, :len(o_y_test[i]), :] = np.eye(len(phone_dict), dtype = 'int8')[o_y_test[i]]
	mask_test[i, :len(o_y_test[i])-20, :] = np.array([[1]*len(phone_dict) for _ in range(len(o_y_test[i])-20)])

_, x_test, _, y_test, _, mask_test = train_test_split(x_test, y_test, mask_test, test_size = 0.1, random_state = 0)

# init parameters
print('set up parameters')
phone_num = len(phone_dict)
layer_num = 2
layer_dim = [512]*layer_num
test_size = x_test.shape[0]
batch_size = 32

x = tf.placeholder(tf.float32, [batch_size, max_length, mfcc_dim])
y = tf.placeholder(tf.float32, [batch_size, max_length, phone_num])
mask = tf.placeholder(tf.float32, [batch_size, max_length, phone_num])
weights = tf.Variable(tf.random_normal([2*layer_dim[-1], phone_num]))
biases = tf.Variable(tf.random_normal([phone_num, ]))

# build model
print('building model')
res = phone_recognizer(x, weights, biases, phone_num, batch_size, layer_num, layer_dim)
mask_res = tf.multiply(res, mask)
tv = tf.trainable_variables()
reg_cost = 1e-5*tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = mask_res, labels = y))+reg_cost
acc_mask = tf.divide(tf.reduce_sum(mask, axis = -1), tf.constant(phone_num, dtype = 'float32'))
predict = tf.argmax(mask_res, 2)
correct_pred = tf.multiply(tf.cast(tf.equal(predict, tf.argmax(y, 2)), tf.float32), acc_mask)
accuracy = tf.divide(tf.reduce_sum(correct_pred), tf.reduce_sum(acc_mask))

def fast_filter(pre):
	for i in range(1, len(pre)-1):
		if pre[i-1] == pre[i+1]:
			pre[i] = pre[i-1]
	return pre

# testing step
print('start testing')
with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint('./models/'))
	acc = 0
	loss = 0
	count = 0
	for c in range(0, test_size, batch_size):
		#print('\r%5.1f'%(100*(c/test_size)), '% ', end = '')
		count += 1
		if c+batch_size > test_size:
			batch_xs = x_test[-1*batch_size:]
			batch_ys = y_test[-1*batch_size:]
			batch_masks = mask_test[-1*batch_size:]
		else:
			batch_xs = x_test[c:c+batch_size]
			batch_ys = y_test[c:c+batch_size]
			batch_masks = mask_test[-1*batch_size:]
		pre = sess.run(predict, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		acc += sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		loss += sess.run(cost, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		
		t = 0.
		s = 0.
		choose = 3
		pre[choose] = fast_filter(pre[choose])
		for i in range(len(pre[choose])):
			if mask_test[choose][i][0] == 0:
				break
			#print(phone_dict[pre[choose][i]], phone_dict[np.argmax(batch_ys, 2)[choose][i]])
			t += 1
			s += 1 if pre[choose][i] == np.argmax(batch_ys, 2)[choose][i] else 0
		print(s/t)
		break
		
	print('\nTest:', 'acc: %5.3f'%(acc/count), ' loss: %.3f'%(loss/count))