from python_speech_features import mfcc
from sphfile import SPHFile
import numpy as np
import librosa
import os

sample_rate = 16000
win_len = 0.025
win_step = 0.01
mfcc_dim = 39
h_window = win_len*sample_rate*0.5#1440*0.5
stride = win_step*sample_rate#400

def get_x(path):
	# window = 400 frames
	# stride = 160 frames
	audio = SPHFile(path).content
	audio = mfcc(audio, numcep = mfcc_dim, nfilt = mfcc_dim, winlen = win_len, winstep = win_step)
	#audio = np.transpose(librosa.feature.mfcc(y = audio.astype(float), sr = sample_rate, n_mfcc = mfcc_dim, n_fft = int(h_window*2), hop_length = int(stride)), [1,0])
	audio = np.array(audio)
	return audio

def get_y(path, length, phone_dict):
	phones = []
	with open(path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			if line == '':
				break
			line = line.split('\n')[0]
			start = int(line.split(' ')[0])
			end = int(line.split(' ')[1])
			phone = line.split(' ')[2]
			if not np.isin([phone], phone_dict)[0]:
				phone_dict = np.append(phone_dict, phone)
			add = np.argwhere(phone_dict == phone)[0][0]
			start = (start-h_window) if start >= h_window else 0
			end = (end-h_window) if end >= h_window else 0
			times = (end//stride)-(start//stride)+(1 if start%stride == 0 else 0)-(1 if end%stride == 0 else 0)
			phones += [add]*int(times)
	while len(phones) > length:
		phones = phones[:-1]
	while len(phones) < length:
		phones += phones[-1:]
	return np.array(phones, dtype = 'int8'), phone_dict


def load_timit(timit_path):
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	phone_dict = np.array([])
	for two in ['train', 'test']:
		here = os.path.join(timit_path, two) 
		file_list = [file for file in os.listdir(here)]
		file_list.sort()
		for file in file_list:
			print('\r%5s'%two, file, end = '')
			here2 = os.path.join(here, file)
			file_list2 = [file for file in os.listdir(here2)]
			file_list2.sort()
			for file2 in file_list2:
				here3 = os.path.join(here2, file2)
				file_list3 = [file.split('.wav')[0] for file in os.listdir(here3) if file.endswith('wav')]
				file_list3.sort()
				for file3 in file_list3:
					here4 = os.path.join(here3, file3)
					audio = get_x(here4+'.wav')
					phones, phone_dict = get_y(here4+'.phn', len(audio), phone_dict)
					if two == 'train':
						x_train.append(audio)
						y_train.append(phones)
					else:
						x_test.append(audio)
						y_test.append(phones)
					if len(phones) != len(audio):
						print('error')
						exit()
	return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), phone_dict

def get_mfcc():
	return np.load('./mfcc/x_train.npy'), np.load('./mfcc/y_train.npy'), np.load('./mfcc/x_test.npy'), np.load('./mfcc/y_test.npy'), np.load('./mfcc/phone_dict.npy')

if __name__ == '__main__':
	x_train, y_train, x_test, y_test, phone_dict = load_timit('../timit/')
	np.save('./