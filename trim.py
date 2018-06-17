import librosa
import numpy as np
from scipy.io.wavfile import read, write
# read
sr, y = read('vctk_sent.wav')
y = y.astype(float)
# split
db = 38
agmax = np.argmax(y)
n = librosa.effects.split(y, top_db = db)
# find the main part
for i, x in enumerate(n):
	if x[0] <= agmax < x[1]:
		c = i
		break
# trim
n = y[n[c][0]:n[c][1]]
# save
write(str(db)+'.wav', sr, n)