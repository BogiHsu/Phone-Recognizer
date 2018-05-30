import os
import numpy as np
import tensorflow as tf
from model import *
from loader import get_mfcc
tf.set_random_seed(0)
np.random.seed(0)

# load names
timit_path = '../timit/'
name_list = {'train':[], 'test':[]}
for two in ['train', 'test']:
		here = os.path.join(timit_path, two) 
		file_list = [file for file in os.listdir(here)]
		file_list.sort()
		for file in file_list:
			here2 = os.path.join(here, file)
			file_list2 = [file for file in os.listdir(here2)]
			file_list2.sort()
			for file2 in file_list2:
				here3 = os.path.join(here2, file2)
				file_list3 = [file.split('.wav')[0] for file in os.listdir(here3) if file.endswith('wav')]
				file_list3.sort()
				for file3 in file_list3:
					here4 = os.path.join(here3, file3)
					name_list[two].append(here4)

# load timit2mfcc data
print('reading data')
o_x_train, o_y_train, o_x_test, o_y_test, phone_dict = get_mfcc()

# preprocessing
print('preprocessing')
max_length = max([len(train) for train in o_y_train]+[len(test) for test in o_y_test])#778
mfcc_dim = 39
filt_silence = False
train_samples = o_x_train.shape[0]
test_samples = o_x_test.shape[0]
x_train = np.zeros([train_samples, max_length, mfcc_dim])
y_train = np.zeros([train_samples, max_length, len(phone_dict)], dtype = 'int8')
mask_train = np.zeros([train_samples, max_length, len(phone_dict)], dtype = 'int8')
for i in range(train_samples):
	x_train[i, :len(o_x_train[i]), :] = o_x_train[i]/60
	y_train[i, :len(o_y_train[i]), :] = np.eye(len(phone_dict), dtype = 'int8')[o_y_train[i]]
	if filt_silence:
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
	else:
		mask_train[i, :len(o_y_train[i]), :] = np.array([[1]*len(phone_dict) for _ in range(len(o_y_train[i]))])

test_samples = o_x_test.shape[0]
x_test = np.zeros([test_samples, max_length, mfcc_dim])
y_test = np.zeros([test_samples, max_length, len(phone_dict)], dtype = 'int8')
mask_test = np.zeros([test_samples, max_length, len(phone_dict)], dtype = 'int8')
for i in range(test_samples):
	x_test[i, :len(o_x_test[i]), :] = o_x_test[i]/60
	y_test[i, :len(o_y_test[i]), :] = np.eye(len(phone_dict), dtype = 'int8')[o_y_test[i]]
	if filt_silence:
		to_many = [0, len(o_y_test[i])]
		past = 0
		for here in np.where(o_y_test[i] == 0)[0]:
			if here-past > 1:
				to_many[0] = past
				to_many[1] = here
				break
			past = here
		to_many[1] = min(len(o_y_test[i]), to_many[1]+1)
		mask_test[i, to_many[0]:to_many[1], :] = np.array([[1]*len(phone_dict) for _ in range(to_many[1]-to_many[0])])
	else:
		mask_test[i, :len(o_y_test[i]), :] = np.array([[1]*len(phone_dict) for _ in range(len(o_y_test[i]))])

# init parameters
print('set up parameters')
phone_num = len(phone_dict)
layer_num = 2
layer_dim = [512]*layer_num
batch_size = 32

x = tf.placeholder(tf.float32, [batch_size, max_length, mfcc_dim])
y = tf.placeholder(tf.float32, [batch_size, max_length, phone_num])
mask = tf.placeholder(tf.float32, [batch_size, max_length, phone_num])
weights = tf.Variable(tf.random_normal([layer_dim[-1], phone_num]))
biases = tf.Variable(tf.random_normal([phone_num, ]))

# for frames <-> time steps
sample_rate = 16000
win_len = 0.025
win_step = 0.01
h_window = win_len*sample_rate*0.5
stride = win_step*sample_rate

# build model
print('building model')
res = build_encoder(x, weights, biases, phone_num, batch_size, False)

# writing step
print('writing')
with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint('./models/'))
	# training data
	f = open('./train_post.csv', 'w')
	for c in range(0, train_samples, batch_size):
		print('\r%5.1f'%(100*(c/train_samples)), '% ', end = '')
		if c+batch_size > train_samples:
			batch_xs = x_train[-1*batch_size:]
			batch_ys = y_train[-1*batch_size:]
			batch_masks = mask_train[-1*batch_size:]
		else:
			batch_xs = x_train[c:c+batch_size]
			batch_ys = y_train[c:c+batch_size]
			batch_masks = mask_train[c:c+batch_size]
		pre = sess.run(res, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		j = 0
		thresh = batch_size if c+batch_size <= train_samples else train_samples-c
		while j < thresh:
			l = np.sum(batch_masks[j, :, 0])
			name = name_list['train'][c+j]
			word_seq_file = open(name+'.wrd', 'r')
			word_seq = word_seq_file.readlines()
			word_seq_file.close()
			
			past = 0
			k = 0
			start = 0
			end = int(word_seq[0].split(' ')[0])
			word = '#h'
			start = (start-h_window) if start >= h_window else 0
			end = (end-h_window) if end >= h_window else 0
			times = int((end//stride)-(start//stride)+(1 if start%stride == 0 else 0)-(1 if end%stride == 0 else 0))
			f.write(name[3:]+',')
			f.write(str(k)+',')
			f.write(word)
			exp_x = np.exp(pre[j, past:past+times, :])
			softmax_x = exp_x/np.repeat(np.sum(exp_x, axis = 1).reshape((-1, 1)), 61, axis = 1)
			for d in softmax_x.reshape((-1,)):
				f.write(',')
				f.write(str(d))
			f.write('\n')
			past += times
			k += 1
			
			for word_info in word_seq:
				start = int(word_info.split(' ')[0])
				end = int(word_info.split(' ')[1])
				word = word_info.split(' ')[2].split('\n')[0]
				start = (start-h_window) if start >= h_window else 0
				end = (end-h_window) if end >= h_window else 0
				times = int((end//stride)-(start//stride)+(1 if start%stride == 0 else 0)-(1 if end%stride == 0 else 0))
				f.write(name[3:]+',')
				f.write(str(k)+',')
				f.write(word)
				exp_x = np.exp(pre[j, past:past+times, :])
				softmax_x = exp_x/np.repeat(np.sum(exp_x, axis = 1).reshape((-1, 1)), 61, axis = 1)
				for d in softmax_x.reshape((-1,)):
					f.write(',')
					f.write(str(d))
				f.write('\n')
				past += times
				k += 1
			end = l
			word = '#h'
			f.write(name[3:]+',')
			f.write(str(k)+',')
			f.write(word)
			exp_x = np.exp(pre[j, past:l, :])
			softmax_x = exp_x/np.repeat(np.sum(exp_x, axis = 1).reshape((-1, 1)), 61, axis = 1)
			for d in softmax_x.reshape((-1,)):
				f.write(',')
				f.write(str(d))
			f.write('\n')
			j += 1
	f.close()
	
	# testing data
	f = open('./test_post.csv', 'w')
	for c in range(0, test_samples, batch_size):
		print('\r%5.1f'%(100*(c/test_samples)), '% ', end = '')
		if c+batch_size > test_samples:
			batch_xs = x_test[-1*batch_size:]
			batch_ys = y_test[-1*batch_size:]
			batch_masks = mask_test[-1*batch_size:]
		else:
			batch_xs = x_test[c:c+batch_size]
			batch_ys = y_test[c:c+batch_size]
			batch_masks = mask_test[c:c+batch_size]
		pre = sess.run(res, feed_dict = {x:batch_xs, y:batch_ys, mask:batch_masks})
		j = 0
		thresh = batch_size if c+batch_size <= test_samples else test_samples-c
		while j < thresh:
			l = np.sum(batch_masks[j, :, 0])
			name = name_list['test'][c+j]
			word_seq_file = open(name+'.wrd', 'r')
			word_seq = word_seq_file.readlines()
			word_seq_file.close()
			
			past = 0
			k = 0
			start = 0
			end = int(word_seq[0].split(' ')[0])
			word = '#h'
			start = (start-h_window) if start >= h_window else 0
			end = (end-h_window) if end >= h_window else 0
			times = int((end//stride)-(start//stride)+(1 if start%stride == 0 else 0)-(1 if end%stride == 0 else 0))
			f.write(name[3:]+',')
			f.write(str(k)+',')
			f.write(word)
			exp_x = np.exp(pre[j, past:past+times, :])
			softmax_x = exp_x/np.repeat(np.sum(exp_x, axis = 1).reshape((-1, 1)), 61, axis = 1)
			for d in softmax_x.reshape((-1,)):
				f.write(',')
				f.write(str(d))
			f.write('\n')
			past += times
			k += 1
			
			for word_info in word_seq:
				start = int(word_info.split(' ')[0])
				end = int(word_info.split(' ')[1])
				word = word_info.split(' ')[2].split('\n')[0]
				start = (start-h_window) if start >= h_window else 0
				end = (end-h_window) if end >= h_window else 0
				times = int((end//stride)-(start//stride)+(1 if start%stride == 0 else 0)-(1 if end%stride == 0 else 0))
				f.write(name[3:]+',')
				f.write(str(k)+',')
				f.write(word)
				exp_x = np.exp(pre[j, past:past+times, :])
				softmax_x = exp_x/np.repeat(np.sum(exp_x, axis = 1).reshape((-1, 1)), 61, axis = 1)
				for d in softmax_x.reshape((-1,)):
					f.write(',')
					f.write(str(d))
				f.write('\n')
				past += times
				k += 1
			end = l
			word = '#h'
			f.write(name[3:]+',')
			f.write(str(k)+',')
			f.write(word)
			exp_x = np.exp(pre[j, past:l, :])
			softmax_x = exp_x/np.repeat(np.sum(exp_x, axis = 1).reshape((-1, 1)), 61, axis = 1)
			for d in softmax_x.reshape((-1,)):
				f.write(',')
				f.write(str(d))
			f.write('\n')
			j += 1
	f.close()
