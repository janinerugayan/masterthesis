import librosa, pickle
import os
import argparse
from os import listdir
from os.path import join

from IPython import embed

import torch
from torch import nn, optim
import torch.nn.functional as F

from vqapc_model import GumbelAPCModel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',   type=str)
parser.add_argument('--sound_file',     type=str)
parser.add_argument('--pretrained_weights',   type=str)
parser.add_argument('--pretrained_VQ',      type=str)
args = parser.parse_args()


'''
    mel spectrogram - 80-dimensional
'''

wav_path = args.sound_file

x, sr = librosa.load(wav_path, sr=44100)
mel_per_wav = librosa.feature.melspectrogram(x, sr=sr, n_mels=80).T

print("for wav file " + wav_path + ", mel spectrogram shape:")
print(mel_per_wav.shape)

n = len(mel_per_wav)
f = open('mel_spectrogram.txt' , 'w')
for i in range(n):
    for item in mel_per_wav[i]:
        f.write(str(item) + ' ')
    f.write('\n')
f.close()


'''
    prepare data - following APC pipeline
'''

max_seq_len = 32000
save_dir = './preprocessed'
utt_id = args.exp_name

id2len = {}
with open('mel_spectrogram.txt', 'r') as f:
    # process the file line by line
    log_mel = []

    for line in f:
        data = line.strip().split()
        log_mel.append([float(i) for i in data])

    id2len[utt_id + '.pt'] = min(len(log_mel), max_seq_len)
    log_mel = torch.FloatTensor(log_mel)  # convert the 2D list to a pytorch tensor
    log_mel = F.pad(log_mel, (0, 0, 0, max_seq_len - log_mel.size(0))) # pad or truncate
    torch.save(log_mel, os.path.join(save_dir, utt_id + '.pt'))

with open(os.path.join(save_dir, 'lengths.pkl'), 'wb') as f:  # sequence lengths to be used for forward function?
    pickle.dump(id2len, f, protocol=4)


'''
    loading pretrained model
'''

pretrained_vqapc = GumbelAPCModel(input_size=80,
                     hidden_size=512,
                     num_layers=3,
                     dropout=0.1,
                     residual=' ',
                     codebook_size=128,
                     code_dim=512,
                     gumbel_temperature=0.5,
                     vq_hidden_size=-1,
                     apply_VQ= [False, False, True]).cuda()

pretrained_vqapc = nn.DataParallel(pretrained_vqapc)

pretrained_weights_path = args.pretrained_weights
pretrained_vqapc.module.load_state_dict(torch.load(pretrained_weights_path))


'''
    using forward method of model class with preprocessed data
'''

with open('./preprocessed/lengths.pkl', 'rb') as f:
    lengths = pickle.load(f)

frames_BxLxM = torch.load('./preprocessed/' + args.exp_name + '.pt')
seq_lengths_B = torch.as_tensor(lengths[args.exp_name + '.pt'], dtype=torch.int64, device=torch.device('cpu'))
testing = True
embed()

predicted_BxLxM, hiddens_NxBxLxH, logits_NxBxLxC = pretrained_vqapc.module.forward(frames_BxLxM, seq_lengths_B, testing)
