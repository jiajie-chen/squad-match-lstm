from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from embedding import UNKNOWN, embeddings
import helpers

vocab, embedding_matrix = embeddings()
vocab_lookup = {word: i for i, word in enumerate(vocab)}
vocab_size, embedding_size = embedding_matrix.shape

def vectorize(text):
    vocab_size = len(vocab_lookup)
    tokens = helpers.tokenize(text)
    return np.array([vocab_lookup.get(token, UNKNOWN) for token in tokens])

class Squad(Net):
    def setup(self):
        # passage = tf.placeholder(tf.int32, [None])
        # question = tf.placeholder(tf.int32, [None])
        embedding = tf.constant(embedding_matrix, name='embedding')
        # embedded_vec = tf.nn.embedding_lookup(embedding, self.input)
