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

from prepare_data import randomseg, CombinedSpeech
from prepare_data import process_wav_multiple, prepare_torch_lengths_multiple

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',   type=str)
parser.add_argument('--sound_file',     type=str)
parser.add_argument('--pretrained_weights',   type=str)
parser.add_argument('--out_path',      type=str)
args = parser.parse_args()



# ---------------------------------------------
#   mel spectrogram - 80-dimensional
# ---------------------------------------------

# wav_path = args.sound_file
# export_dir_path = './preprocessed/'
#
# # randomly segment combined sound file
# min_len = 10000
# max_len = 40000
# randomseg(wav_path, export_dir_path, min_len, max_len)
#
# # process wav files to get log-mel feature vectors
# output_path = export_dir_path + args.exp_name
# process_wav_multiple(export_dir_path, output_path)



# ---------------------------------------------
#   prepare data - following APC pipeline
# ---------------------------------------------

# wav_id = args.exp_name
# logmel_path = export_dir_path
# max_seq_len = 30000
#
# prepare_torch_lengths_multiple(logmel_path, max_seq_len, wav_id)



# ---------------------------------------------
#   loading pretrained model
# ---------------------------------------------

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



# -----------------------------------------------------------------
#   using forward method of model class with preprocessed data
# -----------------------------------------------------------------

dataset = CombinedSpeech('./preprocessed/')
dataset_loader = data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True, drop_last=True)

testing = True

for frames_BxLxM, lengths_B in dataset_loader:
    frames_BxLxM = Variable(frames_BxLxM).cuda()
    lengths_B = Variable(lengths_B).cuda()
    print(frames_BxLxM.size())
    print(lengths_B)
    __, features, __, prevq_rnn_outputs = pretrained_vqapc.module.forward(frames_BxLxM, lengths_B, testing)

print(features.size())
print(features[-1, :, :, :])

print(f'prevq rnn output: {prevq_rnn_outputs}')

prevq = prevq_rnn_outputs.pop().squeeze().cpu().detach().numpy()

# with open(args.out_path + args.exp_name + '.txt', 'w') as file:
#     np.savetext(file, prevq, fmt='%.16f')
