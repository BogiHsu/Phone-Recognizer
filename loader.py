from sklearn.model_selection import train_test_split
from hyperparams import Hyperparams as hp
from python_speech_features import mfcc
from scipy.io import wavfile
from utils import get_spectrograms
import numpy as np
import librosa
import pickle
import sys
import os
np.random.seed(0)

h_window = hp.win_length*0.5
stride = hp.hop_length

def get_x(path):
	# window = 400 frames
	# stride = 160 frames
	#audio, _ = librosa.load(path, mono = True)
	#_, audio = wavfile.read(path)
	#audio = mfcc(audio, samplerate = hp.sr, numcep = hp.mfcc_dim,
	#		nfilt = hp.mfcc_dim, winlen = )p.win_len, winstep = hp.win_step, nfft = 600)
	#audio = pickle.load(open(path.split('.wav')[0]+'.pickle', 'rb'))[2]
	audio, _ = get_spectrograms(path)
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


def load_libri(libri_path):
	x_data = []
	y_data = []
	phone_dict = np.array([])
	here = libri_path
	file_list = [file for file in os.listdir(here)]
	file_list.sort()
	for i, file in enumerate(file_list):
		here2 = os.path.join(here, file)
		file_list2 = [file.split('.wav')[0] for file in os.listdir(here2) if file.endswith('wav')]
		file_list2.sort()
		for j, file2 in enumerate(file_list2):
			print('\r', '%2d'%(i+1), '/', len(file_list), sep = '', end = ' ')
			print('%4d'%(j+1), '/', len(file_list2), sep = '', end = ' ')
			sys.stdout.flush()
			here3 = os.path.join(here2, file2)
			audio = get_x(here3+'.wav')
			phones, phone_dict = get_y(here3+'.phn', len(audio), phone_dict)
			x_data.append(audio)
			y_data.append(phones)
			if len(phones) != len(audio):
				print('error')
				exit()
	print('')
	np.save(hp.xpath, x_data)
	np.save(hp.ypath, y_data)
	np.save(hp.ppath, phone_dict)

def get_mfcc(split = True):
	o_x_data = np.load(hp.xpath)
	o_y_data = np.load(hp.ypath)
	phone_dict = np.load(hp.ppath)
	phone_num = len(phone_dict)
	num_samples = o_x_data.shape[0]
	max_length = max([len(train) for train in o_y_data])
	x_data = np.zeros([num_samples, max_length, hp.mfcc_dim])
	y_data = np.zeros([num_samples, max_length, phone_num], dtype = 'int8')
	mask_data = np.zeros([num_samples, max_length, phone_num], dtype = 'int8')
	for i in range(num_samples):
		x_data[i, :len(o_x_data[i]), :] = o_x_data[i]
		y_data[i, :len(o_y_data[i]), :] = np.eye(phone_num, dtype = 'int8')[o_y_data[i]]
		mask_data[i, :len(o_y_data[i]), :] = np.array([[1]*phone_num for _ in range(len(o_y_data[i]))])
	if split:
		x_train, x_test, y_train, y_test, mask_train, mask_test = train_test_split(x_data, y_data, mask_data, test_size = hp.split, random_state = 0)
		return x_train, x_test, y_train, y_test, mask_train, mask_test, phone_dict, phone_num, max_length
	else:
		return x_data, y_data, mask_data, phone_dict, phone_num, max_length

if __name__ == '__main__':
	load_libri(hp.fpath)
	
