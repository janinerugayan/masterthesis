import librosa, pickle
import os
import argparse
from os import listdir
from os.path import join
import numpy as np
import scipy

from IPython import embed

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable

from vqapc_model import GumbelAPCModel

from prepare_data import process_wav, prepare_torch_lengths, randomseg
from prepare_data import process_wav_multiple, prepare_torch_lengths_multiple
from prepare_data import CombinedSpeech

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
export_dir_path = './preprocessed/'

# randomly segment combined sound file
min_len = 10000
max_len = 60000
randomseg(wav_path, export_dir_path, min_len, max_len)

# process wav files to get log-mel feature vectors
output_path = export_dir_path + args.exp_name
process_wav_multiple(export_dir_path, output_path)


# process_wav(wav_path, out_path)


# x, sr = librosa.load(wav_path, sr=44100)
# mel_per_wav = librosa.feature.melspectrogram(x, sr=sr, n_mels=80).T
#
# print("for wav file " + wav_path + ", mel spectrogram shape:")
# print(mel_per_wav.shape)
#
# n = len(mel_per_wav)
# f = open('mel_spectrogram.txt' , 'w')
# for i in range(n):
#     for item in mel_per_wav[i]:
#         f.write(str(item) + ' ')
#     f.write('\n')
# f.close()


'''
    prepare data - following APC pipeline
'''

wav_id = args.exp_name
logmel_path = export_dir_path
max_seq_len = 1600

prepare_torch_lengths_multiple(logmel_path, max_seq_len, wav_id)


# prepare_torch_lengths(save_dir, utt_id, logmel_path)

# max_seq_len = 32000
# save_dir = './preprocessed'
# utt_id = args.exp_name
#
# id2len = {}
# with open('mel_spectrogram.txt', 'r') as f:
#     # process the file line by line
#     log_mel = []
#
#     for line in f:
#         data = line.strip().split()
#         log_mel.append([float(i) for i in data])
#
#     id2len[utt_id + '.pt'] = min(len(log_mel), max_seq_len)
#     log_mel = torch.FloatTensor(log_mel)  # convert the 2D list to a pytorch tensor
#     log_mel = F.pad(log_mel, (0, 0, 0, max_seq_len - log_mel.size(0))) # pad or truncate
#     torch.save(log_mel, os.path.join(save_dir, utt_id + '.pt'))
#
# with open(os.path.join(save_dir, 'lengths.pkl'), 'wb') as f:  # sequence lengths to be used for forward function?
#     pickle.dump(id2len, f, protocol=4)


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

# seq_lengths_B = []
# with open('./preprocessed/lengths.pkl', 'rb') as f:
#     lengths = pickle.load(f)
# seq_lengths_B = list(lengths.values())
# seq_lengths_B = torch.as_tensor(seq_lengths_B, dtype=torch.int64, device=torch.device('cpu'))

dataset = CombinedSpeech('./preprocessed/')
dataset_loader = data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True, drop_last=True)

testing = True

for frames_BxLxM, lengths_B in dataset_loader:
    frames_BxLxM = Variable(frames_BxLxM[indices_B]).cuda()
    lengths_B = Variable(lengths_B[indices_B]).cuda()
    predicted_BxLxM, hiddens_NxBxLxH, logits_NxBxLxC = pretrained_vqapc.module.forward(frames_BxLxM, lengths_B, testing)
