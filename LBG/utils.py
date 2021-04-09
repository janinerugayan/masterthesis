import librosa
import librosa.display
import scipy
import numpy as np


# feature extraction for the test data
def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)

def process_wav(wav_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=16,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):

    wav, __ = librosa.load(wav_path, sr=sr,
                          offset=offset, duration=duration)

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

    return logmel
