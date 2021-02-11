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
    # return out_path, logmel.shape[-1]
    return logmel.shape[-1]


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


def process_wav_multiple(in_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):

    for file in os.listdir(in_path):
        if file.endswith('.wav'):
            path = in_path + file
            wav, _ = librosa.load(path, sr=sr, offset=offset, duration=duration)
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

            filename = Path(file).stem

            np.save(out_path + '_' + filename + '_logmel.npy', np.transpose(logmel))
