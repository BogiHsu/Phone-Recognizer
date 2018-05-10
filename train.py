import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from model import phone_recognizer
from sklearn.utils import shuffle
from loader import get_mfcc
tf.set_random_seed(0)
np.random.seed(0)

# load timit2mfcc data
print('reading data')
o_x_train, o_y_train, _, o_y_test, phone_dict = get_mfcc()

# preprocessing
print('preprocessing')
num_samples = o_x_train.shape[0]
max_length = max([len(train) for train in o_y_train]+[len(test) for test in o_y_test])
mfcc_dim = 39
x_train = np.zeros([num_samples, max_length, mfcc_dim])
y_train = np.zeros([num_samples, max_length, len(phone_dict)], dtype = 'int8')
mask_train = np.zeros([num_samples, max_length, len(phone_dict)], dtype = 'int8')
for i in range(num_samples):
	x_train[i, :len(o_x_train[i]), :] = o_x_train[i]/60
	y_train[i, :len(o_y_train[i]), :] = np.eye(len(phone_dict), dtype = 'int8')[o_y_train[i]]
	to_many = [0, len(o_y_train[i])]
	past = 0
	for here in np.where(o_y_train[i] == 0)[0]:
		if here-past > 1:
			to_many[0] = past
			to_many[1] = here
			break
		past = here
	to_many[1] = min(len(o_y_train[i]), to_many[1]+1)
	mask_train[i, to_many[0]:to_many[1], :] = np.array([[1]*len(phone_dict) for _ in range(to_many[1]-to_many[0])])
x_train, x_test, y_train, y_test, mask_train, mask_test = train_test_split(x_train, y_train, mask_train, test_size = 0.03, random_state = 0)

# init parameters
print('set up parameters')
phone_num = len(phone_dict)
layer_num = 2
layer_dim = [512]*layer_num
train_size = x_train.shape[0]
test_size = x_test.shape[0]

epochs = 100
v_period = 1
v_size = (test_size)
save_period = 5
max_keep = 10
batch_size = 32
lr = 0.01
dr = 0.35

x = tf.placeholder(tf.float32, [batch_size, max_length, mfcc_dim])
y = tf.placeholder(tf.float32, [batch_size, max_length, phone_num])
mask = tf.placeholder(tf.float32, [batch_size, max_length, phone_num])
weights = tf.Variable(tf.random_normal([2*layer_dim[-1], phone_num]))
biases = tf.Variable(tf.random_normal([phone_num, ]))

# build model
print('building model')
res = phone_recognizer(x, weights, biases, phone_num, batch_size, layer_num, layer_dim, dr)
mask_res = tf.multiply(res, mask)
tv = tf.trainable_variables()
reg1_cost = 3e-6*tf.reduce_sum([tf.nn.l1_loss(v) for v in tv ])
reg2_cost = 7e-6*tf.reduce_sum([tf.nn.l2_loss(v) for v in tv ])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = mask_res, labels = y))+reg1_cost+reg2_cost
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
acc_mask = tf.divide(tf.reduce_sum(mask, axis = -1), tf.constant(phone_num, dtype = 'float32'))
predict = tf.argmax(res, 2)
correct_pred = tf.multiply(tf.cast(tf.equal(predict, tf.argmax(y, 2)), tf.float32), acc_mask)
accuracy = tf.divide(tf.reduce_sum(correct_pred), tf.reduce_sum(acc_mask))

# training step
print('start training')
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
saver = tf.train.Saver(max_to_keep = max_keep)
his = open('./history-adam37', 'w')
with tf.Session() as sess:
	sess.run(init[0])
	sess.run(init[1])
	step = 1
	while step <= epochs:
		count = 0
		t_acc = 0
		t_loss = 0
		for c in range(0, len(x_train), batch_size):
			count += 1
			if c+batch_size > len(x_train):
				batch_xs = x_train[-1*batch_size:]
				batch_ys = y_train[-1*batch_size:]
				batch_masks = mask_train[-1*batch_size:]
			else:
				batch_xs = x_train[c:c+batch_size]
				batch_ys = y_train[c:c+batch_size]
				batch_masks = mask_train[c:c+batch_size]
			_, tmpa, tmpc = sess.run([train_op, accuracy, cost], feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
			t_acc += tmpa
			t_loss += tmpc
			print('\rEpoch: ', '%3d/%3d'%(step, epochs), ' | Train: ', sep = '', end = '')
			print('%5.1f'%(100*(c/train_size)), '% ', sep = '', end = '')
			print(' acc: %5.3f'%(t_acc/count), ' loss: %5.3f'%(t_loss/count), end = '')
		x_train, y_train, mask_train = shuffle(x_train, y_train, mask_train, random_state = step)
		
		if (step%v_period == 0 or step%save_period == 0):
			acc = 0
			loss = 0
			count = 0
			for c in range(0, v_size, batch_size):
				count += 1
				if c+batch_size > v_size:
					batch_xs = x_test[-1*batch_size:]
					batch_ys = y_test[-1*batch_size:]
					batch_masks = mask_test[-1*batch_size:]
				else:
					batch_xs = x_test[c:c+batch_size]
					batch_ys = y_test[c:c+batch_size]
					batch_masks = mask_test[c:c+batch_size]
				acc += sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
				loss += sess.run(cost, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
			if step%v_period == 0:
				his.write('step:'+str(step)+' acc: %5.3f'%(acc/count)+' loss: %.3f\n'%(loss/count))
				his.flush()
				print(' | Val:', 'acc: %5.3f'%(acc/count), ' loss: %.3f'%(loss/count), end = '')
			if step%save_period == 0:
				save_path = saver.save(sess, 'models/model-'+str(step).zfill(3)+'-'+str(round(acc/count, 2))+'.ckpt')
				print('\nsaving model to %s'%save_path, end = '')
		print('')
		step += 1
his.close()
