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



def process_wav_multiple(in_path, out_path, sr=16000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=16,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    # original: sr=160000, n_fft=2048, hop_length=160, win_length=400
    for file in os.listdir(in_path):
        if file.endswith('.wav'):
            path = in_path + file
            wav, _ = librosa.load(path, sr=sr)  # removed offset and duration, because individual wav files are processed
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

            np.save(out_path + filename + '_logmel.npy', np.transpose(logmel))
            with open(out_path + filename + '_logmel.txt', 'w') as file:
                np.savetxt(file, np.transpose(logmel), fmt='%.6f')
