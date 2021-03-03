import numpy as np
import torch.nn as nn

# embedding_dir = './results/embedding_from_training/'
#
# embedding = np.load(embedding_dir + 'embedding_epoch_1.npy')
#
# print(embedding.shape)

codebook_size = 128
code_dim = 512


codebook_CxE = nn.Linear(codebook_size, code_dim, bias=False)
embedding = codebook_CxE.cpu().numpy()
