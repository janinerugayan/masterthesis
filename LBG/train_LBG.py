import numpy as np
import argparse
import lbg_algo_ver1 as lbg1
import lbg_algo_ver2 as lbg2
import torch

from datasets import LibriSpeech
from torch.utils import data



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

    # generating codebook
    for frames_BxLxM, lengths_B in train_data_loader:
        _, indices_B = torch.sort(lengths_B, descending=True)

        frames_arr = frames_BxLxM[0].numpy().squeeze()

        for i in range(1, config.batch_size):
            frames = frames_BxLxM[i].numpy().squeeze()
            frames_arr = np.vstack((frames_arr, frames))

        print(np.shape(frames_arr))


        vq_lg = lbg2.VQ_LGB(frames,cb_size,0.00005,3000)
        vq_lg.run()
        codebook2 = vq_lg.get_codebook()

        print('CODEBOOK 2:')
        print(np.shape(codebook2))


        break

main()
