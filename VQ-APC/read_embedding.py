import numpy as np

embedding_dir = './results/embedding_from_training/'

embedding = np.load(embedding_dir + 'embedding_epoch_1.npy')

print(embedding.shape)
