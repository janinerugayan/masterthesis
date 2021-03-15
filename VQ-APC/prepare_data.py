import librosa
import pickle
import scipy
import numpy as np
import torch
import torch.nn.functional as F
import os
from pydub import AudioSegment
import argparse
import random
from pathlib import Path
from torch.utils import data


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def process_wav(wav_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    # wav = mulaw_encode(wav, mu=2**bits)

    # np.save(out_path + '.wav.npy', wav)
    np.save(out_path + '_logmel.npy', np.transpose(logmel))



def prepare_torch_lengths(save_dir, utt_id, logmel_path):

    data = np.load(logmel_path)

    id2len = {}
    log_mel = []

    for row in range(len(data)):
        feature_vector = data[row]
        log_mel.append([float(i) for i in feature_vector])

    id2len[utt_id + '.pt'] = len(log_mel)
    log_mel = torch.FloatTensor(log_mel)  # convert 2D list to a pytorch as_tensor
    # log_mel = pad(log_mel, (0, 0, 0, max_seq_len - log_mel.size(0))) # pad or truncate
    torch.save(log_mel, os.path.join(save_dir, utt_id + '.pt'))

    with open(os.path.join(save_dir, 'lengths.pkl'), 'wb') as f:  # sequence lengths to be used for forward function?
        pickle.dump(id2len, f, protocol=4)



# -----------------------------------------------------
# for segmenting into multiple wav files and processing
# -----------------------------------------------------


def randomseg(wav_path, export_dir_path, min_len, max_len):

    wav_original = AudioSegment.from_wav(wav_path)

    total_len = len(wav_original)

    t_start = 0
    count = 0

    while True:
        rand_duration = random.randint(min_len, max_len)

        if t_start + rand_duration >= total_len:
            wav_segment = wav_original[t_start : total_len]
            wav_segment.export(export_dir_path + str(count) + '.wav', format="wav")
            break

        wav_segment = wav_original[t_start : t_start + rand_duration]
        wav_segment.export(export_dir_path + str(count) + '.wav', format="wav")
        count += 1
        t_start = t_start + rand_duration
        break # remove later, this is just for testing 1 segment of speech


def process_wav_multiple(in_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):

    for file in os.listdir(in_path):
        if file.endswith('.wav'):
            path = in_path + file
            wav, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
            wav = wav / np.abs(wav).max() * 0.999
            mel = librosa.feature.melspectrogram(wav,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 win_length=win_length,
                                                 fmin=fmin,
                                                 power=1)
            logmel = librosa.amplitude_to_db(mel, top_db=top_db)
            logmel = logmel / top_db + 1

            filename = Path(file).stem

            np.save(out_path + filename + '_logmel.npy', np.transpose(logmel))
            with open(out_path + filename + '_logmel.txt', 'w') as file:
                np.savetxt(file, np.transpose(logmel), fmt='%.6f')


def prepare_torch_lengths_multiple(logmel_path, max_seq_len, wav_id):

    id2len = {}

    for file in os.listdir(logmel_path):
        log_mel = []
        if file.endswith('.npy'):
            filename = Path(file).stem
            data = np.load(logmel_path + file)
            for row in range(len(data)):
                log_mel.append([float(i) for i in data[row]])
            id2len[filename + '.pt'] = min(len(log_mel), max_seq_len)
            # id2len[filename + '.pt'] = len(log_mel)
            log_mel = torch.FloatTensor(log_mel)  # convert 2D list to a pytorch tensor
            log_mel = F.pad(log_mel, (0, 0, 0, max_seq_len - log_mel.size(0))) # pad or truncate
            torch.save(log_mel, os.path.join(logmel_path, filename + '.pt'))
            print(f'file: {filename} torch size: {log_mel.size()}')

    with open(os.path.join(logmel_path, 'lengths.pkl'), 'wb') as f:  # sequence lengths to be used for forward function?
        pickle.dump(id2len, f, protocol=4)



# -----------------------------------------------------
# for loading the combined speech data
# -----------------------------------------------------


# taken from APC datasets.py
class CombinedSpeech(data.Dataset):
  def __init__(self, path):
    self.path = path
    self.ids = [f for f in os.listdir(self.path) if f.endswith('.pt')]
    with open(os.path.join(path, 'lengths.pkl'), 'rb') as f:
      self.lengths = pickle.load(f)

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, index):
    x = torch.load(os.path.join(self.path, self.ids[index]))
    l = self.lengths[self.ids[index]]
    return x, l

class LoadSpeechSegment(data.Dataset):
    def __init__(self, path, file):
        self.path = path
        self.id = [file]
        with open(os.path.join(path, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        x = torch.load(os.path.join(self.path, self.id[index]))
        l = self.lengths[self.id[index]]
        return x, l
