import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from utils import *
from model import *
from loader import get_mfcc
from hyperparams import Hyperparams as hp
np.random.seed(0)
tf.set_random_seed(0)


# load mfcc data
print('reading data')
x_data, _, mask_data, phone_dict, phone_num, max_length = get_mfcc(False)
n_phone_dict = np.load('mfcc/phone_dict_lb.npy')
n_phone_dict = np.concatenate((n_phone_dict, np.array(['sil'])))
n_phone_dict = np.concatenate((n_phone_dict, np.array(['OY2'])))
transfer = [np.where(phone_dict == e)[0][0] for e in n_phone_dict
			if (len(np.where(phone_dict == e)[0])) != 0]

# init parameters
print('set up parameters')
data_size = x_data.shape[0]
x = tf.placeholder(tf.float32, [hp.batch_size, max_length, hp.mfcc_dim])
mask = tf.placeholder(tf.float32, [hp.batch_size, max_length, phone_num])

files = []
here = hp.fpath
file_list = [file for file in os.listdir(here)]
file_list.sort()
for i, file in enumerate(file_list):
	here2 = os.path.join(here, file)
	file_list2 = [file.split('.wav')[0] for file in os.listdir(here2) if file.endswith('wav')]
	file_list2.sort()
	files += [os.path.join(here2, name) for name in file_list2]

# build model
print('building model')
res = build_encoder(x, phone_num, False, True)
mask_res = tf.multiply(res, mask)

# testing step
print('start testing')
with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, tf.train.latest_checkpoint(hp.mpath))
	for c in range(0, data_size, hp.batch_size):
		print('%4d/%4d'%(c+1, data_size), end = '\r')
		sys.stdout.flush()
		if c+hp.batch_size > data_size:
			batch_xs = x_data[-1*hp.batch_size:]
			batch_masks = mask_data[-1*hp.batch_size:]
			c = data_size-hp.batch_size
		else:
			batch_xs = x_data[c:c+hp.batch_size]
			batch_masks = mask_data[c:c+hp.batch_size]
		pre = sess.run(mask_res, feed_dict = {x:batch_xs, mask:batch_masks})
		for i in range(hp.batch_size):
			name = files[c+i]
			f = open(name+'.pickle', 'wb')
			l = np.sum(batch_masks[i, :, 0])
			mel, mag = get_spectrograms(name+'.wav')
			td = np.array([pre[i][j][transfer] for j in range(l)])
			while td.shape[1] < len(n_phone_dict):
				td = np.concatenate((td, np.zeros((td.shape[0], 1))), axis = -1)
			pickle.dump([batch_xs[i, :l, :], td, mel, mag], f)
			f.close()
