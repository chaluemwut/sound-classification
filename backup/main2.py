import torchaudio, random, torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import IPython.display as ipd
from torchaudio import transforms

def load_audio(file_name):
    data,sr = torchaudio.load(f"data/{file_name}.wav")
    return (data, sr)

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

def process():
  for i in range(1, 2):
    data1, sr1 = load_audio(f"1_{i}")
    data = (data1, sr1)
    r_data = rechannel(data, 2)
    p_data = pad_trunc(r_data, 4000)
    spec = transforms.MelSpectrogram(sr1)(data1)
    spec = transforms.AmplitudeToDB(top_db=80)(spec)
    aug_sgram = spectro_augment(spec)
    # plot_waveform(data1, sr1)
    print(aug_sgram)


def rechannel(aud, new_channel):
  sig, sr = aud

  if (sig.shape[0] == new_channel):
      # Nothing to do
    return aud

  if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
    resig = sig[:1, :]
  else:
      # Convert from mono to stereo by duplicating the first channel
    resig = torch.cat([sig, sig])

  return ((resig, sr))

def pad_trunc(aud, max_ms):
  sig, sr = aud
  num_rows, sig_len = sig.shape
  max_len = sr//1000 * max_ms
  print(max_len)
  if sig_len > max_len:
    sig = sig[:, :max_len]
  else:
    pad_begin_len  = random.randint(0, max_len - sig_len)
    pad_end_len = max_len - sig_len - pad_begin_len

    pad_begin = torch.zeros((num_rows, pad_begin_len))
    pad_end = torch.zeros((num_rows, pad_end_len))
    sig = torch.cat((pad_begin, sig, pad_end), 1)
  
  return (sig, sr)

def time_shift(aud, shift_limit):
  sig,sr = aud
  _, sig_len = sig.shape
  shift_amt = int(random.random() * shift_limit * sig_len)
  return (sig.roll(shift_amt), sr)
  
def mel(sig, sr):
  transform = torchaudio.transforms.MelSpectrogram(sr)
  spec = transform(sig)
  top_db = 80
  spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
  print(spec.shape)
  return spec

def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
  _, n_mels, n_steps = spec.shape
  mask_value = spec.mean()
  aug_spec = spec

  freq_mask_param = max_mask_pct * n_mels
  for _ in range(n_freq_masks):
    aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

  time_mask_param = max_mask_pct * n_steps
  for _ in range(n_time_masks):
    aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

  return aug_spec

process()
