# !pip install scipy==1.13.0
# !pip install librosa
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from scipy.io import wavfile
# import IPython.display 
# from IPython.display import Audio, display
import IPython

y, sr = librosa.load(librosa.ex('trumpet'))
# y, sr = librosa.load('data.wav')
# print(sr)
fs, data = wavfile.read('data.wav')

S = np.abs(librosa.stft(y))
y_inv = librosa.griffinlim(S)
y_istft = librosa.istft(S)

IPython.display.Audio(y, rate=sr)
# display(Audio(y,rate=sr))

# ipd.display(ipd.Audio(y,rate=sr))

# print(y)

