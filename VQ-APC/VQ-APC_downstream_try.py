import librosa, pickle
import os
import argparse
from os import listdir
from os.path import join
import numpy as np
import scipy
from pathlib import Path

from IPython import embed

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable

from vqapc_model import GumbelAPCModel

from prepare_data import randomseg, CombinedSpeech, LoadSpeechSegment
from prepare_data import process_wav_multiple, prepare_torch_lengths_multiple

from phoneseg_algorithms import l2_segmentation

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',   type=str)
parser.add_argument('--sound_file',     type=str)
parser.add_argument('--pretrained_weights',   type=str)
parser.add_argument('--embedding',     type=str)
parser.add_argument('--out_path',      type=str)
args = parser.parse_args()



# ---------------------------------------------
#   mel spectrogram - 80-dimensional
# ---------------------------------------------

# wav_path = args.sound_file
# export_dir_path = './preprocessed/'
#
# # randomly segment combined sound file
# min_len = 1500
# max_len = 1700
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
# max_seq_len = 1600
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

print(pretrained_vqapc)

# get VQ layer codebook
vq_layer = pretrained_vqapc.module.vq_layers.codebook_CxE
print(vq_layer)


# -----------------------------------------------------------------
#   using forward method of model class with preprocessed data
# -----------------------------------------------------------------

# logmel_path = './preprocessed/'
#
# for file in os.listdir(logmel_path):
#
#     features = []
#     prevq_rnn_outputs = []
#
#     if file.endswith('.pt'):
#
#         print(f'VQ-APC working on: {file}')
#
#         filename = Path(file).stem
#
#         dataset = LoadSpeechSegment(logmel_path, file)
#         dataset_loader = data.DataLoader(dataset, batch_size=1, num_workers=8, shuffle=True, drop_last=True)
#
#         testing = True
#
#         for frames_BxLxM, lengths_B in dataset_loader:
#             frames_BxLxM = Variable(frames_BxLxM).cuda()
#             lengths_B = Variable(lengths_B).cuda()
#             __, features, __, prevq_rnn_outputs, __ = pretrained_vqapc.module.forward(frames_BxLxM, lengths_B, testing)
#
#         prevq = prevq_rnn_outputs.pop().squeeze().cpu().detach().numpy()
#
#         with open(args.out_path + 'logmel/' + filename + '.txt', 'w') as file:
#             np.savetxt(file, prevq, fmt='%.16f')



# -----------------------------------------------------------------
#   phone segmentation from VQ-seg code of Herman Kamper
# -----------------------------------------------------------------

# read pre-quantisation
prevq_path = args.out_path + 'logmel/'
prevq_dict = {}
for file in os.listdir(prevq_path):
    if file.endswith('.txt'):
        filename = Path(file).stem
        prevq_dict[filename] = np.loadtxt(prevq_path + file)


# read embedding matrix

codebook_from_training = np.load(args.embedding)
embedding = np.transpose(codebook_from_training)
print(f'Embedding matrix shape: {embedding.shape}')
print(embedding)


# segmentation
boundaries_dict = {}
code_indices_dict = {}

# using phoneseg algorithm: L2 Segmentation
n_min_frames = 0
n_max_frames = 15
dur_weight = 400  # 20**2

for utt_key in prevq_dict:
    z = prevq_dict[utt_key]
    if z.ndim == 1:
        continue
    print(f'Performing phone segmentation on {utt_key}')
    boundaries, code_indices = l2_segmentation(embedding, z, n_min_frames,
                                n_max_frames, dur_weight)
    # do we need to upsample it? was it downsampled in the first place?
    boundaries_dict[utt_key] = boundaries
    code_indices_dict[utt_key] = code_indices

# write code indices
output_dir = args.out_path + 'indices/'
for utt_key in code_indices_dict:
    np.save(output_dir + utt_key + '_indices.npy', np.array([i[-1] for i in code_indices_dict[utt_key]], dtype=np.int))

# write boundaries
output_dir = args.out_path + 'boundaries/'
for utt_key in boundaries_dict:
    np.save(output_dir + utt_key + '_boundaries.npy', np.array(boundaries_dict[utt_key], dtype=np.bool))

# write intervals
output_dir = args.out_path + 'intervals/'
for utt_key in code_indices_dict:
    with open(output_dir + utt_key + '_intervals.txt', 'w') as f:
        for start, end, index in code_indices_dict[utt_key]:
            f.write("{:d} {:d} {:d}\n".format(start, end, index))
