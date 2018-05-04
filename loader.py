from python_speech_features import mfcc
from sphfile import SPHFile
import os
import numpy as np

def timit2mfcc(timit_path):
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
				file_list3 = [file for file in os.listdir(here3) if file.endswith('wav')]
				file_list3.sort()
				for file3 in file_list3:
					here3 = os.path.join(here3, file3)
					audio = SPHFile(here3).content
					print(audio)
					exit()

timit2mfcc('../timit/')
