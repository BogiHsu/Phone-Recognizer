''' ref: https://www.github.com/kyubyong/dc_tts '''
import os
import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import librosa
from copy import deepcopy

from hyperparams import Hyperparams as hp

def get_spectrograms(fpath):
	'''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
	Args:
	  sound_file: A string. The full path of a sound file.

	Returns:
	  mel: A 2d array of shape (T, n_mels) <- Transposed
	  mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
	'''

	# Loading sound file
	#y, sr = librosa.load(fpath, sr=hp.sr)
	_, y = wavfile.read(fpath)
	y = y.astype(float)

	# Trimming
	if hp.is_trimming:
		y, _ = librosa.effects.trim(y)

	# Preemphasis
	y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

	# stft
	linear = librosa.stft(y=y,
						  n_fft=hp.n_fft,
						  hop_length=hp.hop_length,
						  win_length=hp.win_length)

	# magnitude spectrogram
	mag = np.abs(linear)  # (1+n_fft//2, T)

	# mel spectrogram
	mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
	mel = np.dot(mel_basis, mag)  # (n_mels, t)

	# to decibel
	mel = 20 * np.log10(np.maximum(1e-5, mel))
	mag = 20 * np.log10(np.maximum(1e-5, mag))

	# normalize
	mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
	mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

	# Transpose
	mel = mel.T.astype(np.float32)  # (T, n_mels)
	mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

	return mel, mag


def spectrogram2wav(mag):
	'''# Generate wave file from spectrogram'''
	# transpose
	mag = mag.T

	# de-noramlize
	mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

	# to amplitude
	mag = np.power(10.0, mag * 0.05)

	# wav reconstruction
	wav = griffin_lim(mag)

	# de-preemphasis
	wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

	# Trimming
	if hp.is_trimming:
		wav, _ = librosa.effects.trim(wav)

	return wav.astype(np.float32)


def griffin_lim(spectrogram):
	'''Applies Griffin-Lim's raw.
	'''
	X_best = deepcopy(spectrogram)
	for i in range(hp.n_iter):
		X_t = invert_spectrogram(X_best)
		est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
		phase = est / np.maximum(1e-8, np.abs(est))
		X_best = spectrogram * phase
	X_t = invert_spectrogram(X_best)
	y = np.real(X_t)

	return y

def invert_spectrogram(spectrogram):
	'''
	spectrogram: [f, t]
	'''
	return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")
