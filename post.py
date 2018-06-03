import os
import numpy as np
import tensorflow as tf
from model import *
from python_speech_features import mfcc
from hyperparams import Hyperparams as hp
tf.set_random_seed(0)
np.random.seed(0)

class PR():
	def __init__(self):
		self.model_path = './models/'
		self.phone_num = 70
		self.max_length = 1924
		self.mfcc_dim = 39
		self.sample_rate = 16000
		self.win_len = 0.025
		self.win_step = 0.01
		self.h_window = win_len*sample_rate*0.5
		self.stride = win_step*sample_rate
		layer_num = 2
		layer_dim = [512]*layer_num

		self.x = tf.placeholder(tf.float32, [batch_size, max_length, mfcc_dim])
		self.res = build_encoder(x, phone_num, batch_size, layer_dim, False, True)
		
		self.sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
	
	def __del__(self):
		self.sess.close()

	def get_x(self, audio):
		audio = mfcc(audio, samplerate = self.sample_rate,
				numcep = self.mfcc_dim, nfilt = self.mfcc_dim,
				winlen = self.win_len, winstep = self.win_step)
		audio = np.array(audio)/60
		return audio
	
	def predict(self, batch_xs):
		return self.sess.run(self.res, feed_dict = {x:batch_xs})