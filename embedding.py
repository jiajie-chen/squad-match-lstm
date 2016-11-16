# this generates embedding cache files (cache/embedding.npy and cache/vocab.json) that are loaded by squad.py

import numpy as np
import json

UNKNOWN = "*UNKNOWN*"

def load_embedding(filename):
    embeddings = {}
    for line in open(filename):
        parts = line.split()
        embeddings[parts[0]] = map(float, parts[1:])
    embedding_size = len(embeddings.values()[0])
    embeddings[UNKNOWN] = [0.0 for _ in xrange(embedding_size)]
    
    words = embeddings.keys()
    embedding_matrix = np.array([embeddings[word] for word in embeddings.iterkeys()])
    return words, embedding_matrix

def generate_embedding_files(filename):
    vocab, mat = load_embedding(filename)
    np.save(open('cache/embedding.npy', 'w'), mat)
    json.dump(vocab, open('cache/vocab.json', 'w'))

def embeddings():
    words = json.load(open('cache/vocab.json'))
    matrix = np.load(open('cache/embedding.npy'))
    return words, matrix

if __name__ == '__main__':
    generate_embedding_files('data/glove.6B.50d.txt')
