import numpy as np
import argparse
import lbg_algo_ver1 as lbg1
import lbg_algo_ver2 as lbg2
import torch
import os

from datasets import LibriSpeech
from torch.utils import data
from scipy.spatial import distance
from os import listdir

from utils import process_wav_multiple, randomseg



def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--learning_rate", type=float)
    # parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--librispeech_home", type=str)
    parser.add_argument("--train_partition", nargs="+", required=True)
    parser.add_argument("--train_sampling", default=1., type=float)
    parser.add_argument("--val_partition", nargs="+", required=True)
    parser.add_argument("--val_sampling", default=1., type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--load_data_workers", default=8, type=int)
    parser.add_argument("--codebook_size", type=int)
    parser.add_argument("--sound_file", type=str)
    parser.add_argument("--store_path", type=str)
    parser.add_argument("--exp_name", type=str)
    config = parser.parse_args()


    # define data loaders
    train_set = LibriSpeech(home=config.librispeech_home,
                            partition=config.train_partition,
                            sampling=config.train_sampling)
    train_data_loader = data.DataLoader(train_set, batch_size=config.batch_size,
                                        num_workers=config.load_data_workers,
                                        shuffle=True)

    val_set = LibriSpeech(home=config.librispeech_home,
                            partition=config.val_partition,
                            sampling=config.val_sampling)
    val_data_loader = data.DataLoader(val_set, batch_size=config.batch_size,
                                        num_workers=config.load_data_workers,
                                        shuffle=True)

    cb_size = config.codebook_size

    # ---------------------------------------------
    #   LBG for generating the codebook
    # ---------------------------------------------

    for frames_BxLxM, lengths_B in train_data_loader:
        _, indices_B = torch.sort(lengths_B, descending=True)

        frames_arr = frames_BxLxM[0].numpy().squeeze()

        for i in range(1, config.batch_size):
            frames = frames_BxLxM[i].numpy().squeeze()
            frames_arr = np.vstack((frames_arr, frames))

        print(np.shape(frames_arr))

        # codebook1, __, __ = lbg1.generate_codebook(frames_arr, cb_size)
        #
        # print('CODEBOOK 1:')
        # print(np.shape(codebook1))

        vq_lg = lbg2.VQ_LGB(frames,cb_size,0.00005,3000)
        vq_lg.run()
        codebook2 = vq_lg.get_codebook()

        print('CODEBOOK 2:')
        print(np.shape(codebook2))

        break


    # ---------------------------------------------
    #   mel spectrogram - 80-dimensional
    # ---------------------------------------------

    wav_path = config.sound_file
    export_dir_path = config.store_path + config.exp_name + '/'
    os.mkdir(export_dir_path)

    # randomly segment combined sound file
    min_len = 2000  # 1999 for numbers 0-9 test case
    max_len = 2100
    randomseg(wav_path, export_dir_path, min_len, max_len)

    # process wav files to get log-mel feature vectors
    in_path = export_dir_path
    out_path = export_dir_path
    process_wav_multiple(in_path, out_path)


    # ---------------------------------------------
    #   VQ - code assignments using LBG codebook
    # ---------------------------------------------

    for file in os.listdir(export_dir_path):

        codes = []

        if file.endswith('_logmel.npy'):
            logmel = np.load(export_dir_path + file)
            print(np.shape(logmel))

            distances = distance.cdist(logmel, codebook2, metric="sqeuclidean")
            print(np.shape(distances))

            codes = np.argmin(distances, axis=1)
            print(np.shape(codes))


main()
