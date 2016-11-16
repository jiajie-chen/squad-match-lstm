from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from embedding import UNKNOWN, embeddings

vocab, embedding_matrix = embeddings()
vocab_lookup = {word: i for i, word in enumerate(vocab)}
vocab_size, embedding_size = embedding_matrix.shape

class Squad(Net):
    def setup(self):
        pass

print embedding_matrix[vocab_lookup['space']]
