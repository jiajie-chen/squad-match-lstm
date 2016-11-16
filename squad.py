from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from embedding import UNKNOWN, embeddings
from helpers import tokenize
import dataset
from util import rand_variable, weight_var
import random

vocab, embedding_matrix = embeddings()
vocab_lookup = {word: i for i, word in enumerate(vocab)}
vocab_size, embedding_size = embedding_matrix.shape

passage_max_length = 200
question_max_length = 20

def vectorize(text, fixed_length=None):
    vocab_size = len(vocab_lookup)
    tokens = tokenize(text)
    if fixed_length is not None:
        tokens = (tokens + [0] * max(0, fixed_length - len(tokens)))[:fixed_length]
    return np.array([vocab_lookup.get(token, vocab_lookup[UNKNOWN]) for token in tokens])

def output_mask(passage, answers):
    # returns a vector -- same length as the passage -- with 1 if the token is part of an answer, otherwise 0
    answer_marker = "$$answer$" # hack !! !!! !  !  yikes ! ! ! ! !
    for answer in answers:
        replacement = " ".join([answer_marker + w for w in tokenize(answer)])
        passage = passage.replace(answer, replacement)
    return np.array([(1 if token.startswith(answer_marker) else 0) for token in tokenize(passage)])

def make_fixed_length(a, length):
    return np.concatenate([a, [0] * (length - len(a))]) if len(a) < length else a[:length]

def vectors_from_question(para, qa):
    # return an (input, output) tuple, where input is a tuple (passage, question) and output is an output mask
    passage = vectorize(para.passage, fixed_length=passage_max_length)
    # print para.passage
    question = vectorize(qa.question, fixed_length=question_max_length)
    # print qa.question
    # print [answer['text'] for answer in qa.answers]
    mask = output_mask(para.passage, [answer['text'] for answer in qa.answers])
    mask = make_fixed_length(mask, passage_max_length)
    return (passage, question), mask

def questions_from_dataset(ds):
    for para in ds.paragraphs:
        for qa in para.qas:
            yield (para, qa)

class Squad(Net):
    def setup(self):
        passage = tf.placeholder(tf.int32, [None, passage_max_length], name='passage')
        question = tf.placeholder(tf.int32, [None, question_max_length], name='question')
        desired_output = tf.placeholder(tf.float32, [None, passage_max_length], name='desired_output')
        
        embedding = tf.constant(embedding_matrix, name='embedding', dtype=tf.float32)
        question_embedded = tf.nn.embedding_lookup(embedding, question)
        passage_embedded = tf.nn.embedding_lookup(embedding, passage)
        
        def create_dense(input, input_size, output_size, relu=True):
            weights = weight_var([input_size, output_size])
            biases = weight_var([output_size])
            r = tf.matmul(input, weights) + biases
            return tf.nn.relu(r) if relu else r
        
        dropout = tf.placeholder(tf.float32)
        
        with tf.variable_scope('question_lstm'):
            question_cell = tf.nn.rnn_cell.LSTMCell(embedding_size)
            question_cell = tf.nn.rnn_cell.DropoutWrapper(question_cell, output_keep_prob=dropout)
            question_cell = tf.nn.rnn_cell.MultiRNNCell([question_cell] * 2)
            question_vecs, _ = tf.nn.dynamic_rnn(question_cell, question_embedded, dtype=tf.float32)
        
        question_vecs = tf.transpose(question_vecs, [1, 0, 2])
        question_vec = tf.gather(question_vecs, int(question_vecs.get_shape()[0]) - 1) # shape is (batch_size, embedding_size)
        question_vec = tf.reshape(question_vec, (-1, 1, embedding_size))
        
        # for each token vector in the passage, concatenate the question vec onto the end
        question_vec = tf.tile(question_vec, [1, passage_max_length, 1])
        passage_with_question = tf.concat(2, [passage_embedded, question_vec])
        
        with tf.variable_scope('passage_lstm'):
            passage_cell = tf.nn.rnn_cell.LSTMCell(embedding_size * 2)
            passage_cell = tf.nn.rnn_cell.DropoutWrapper(passage_cell, output_keep_prob=dropout)
            passage_cell = tf.nn.rnn_cell.MultiRNNCell([passage_cell] * 2)
            sequence_labels, _ = tf.nn.dynamic_rnn(passage_cell, passage_with_question, dtype=tf.float32)
        
        sequence_label_size = embedding_size * 2
        seq_reshaped = tf.reshape(sequence_labels, (-1, sequence_label_size))
        output_reshaped = create_dense(seq_reshaped, sequence_label_size, 1)
        output = tf.reshape(output_reshaped, (-1, passage_max_length))
        
        # output = tf.Print(output, [output])
        loss = tf.reduce_mean(tf.reduce_sum(tf.pow(desired_output - output, 2), reduction_indices=[1]))
        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
        
        self.passage = passage
        self.question = question
        self.output = output
        self.desired_output = desired_output
        self.train_step = train_step
        self.loss = loss
        self.dropout = dropout
    
    def train(self, paragraph_question_pairs):
        vectors = [vectors_from_question(p, q) for p, q in paragraph_question_pairs]
        # print vectors[0]
        questions = np.array([question for ((passage, question), mask) in vectors])
        passages = np.array([passage for ((passage, question), mask) in vectors])
        masks = np.array([mask for ((passage, question), mask) in vectors])
        
        feed = {self.passage: passages, self.question: questions, self.desired_output: masks, self.dropout: 0.5}
        _, loss = self.session.run([self.train_step, self.loss], feed_dict=feed)
        print loss

def iterate_batches(list, size=10):
    i = 0
    while True:
        yield [list[i+j] for j in range(size)]
        i += size

def train():
    n = Squad(dir_path='save/squad1')
    questions = list(questions_from_dataset(dataset.train()))
    random.shuffle(questions)
    test_questions = list(questions_from_dataset(dataset.test()))
    random.shuffle(test_questions)
    
    i = 0
    for i, batch in enumerate(iterate_batches(questions, size=20)):
        n.train(batch)
        if i % 10 == 0:
            n.save()


if __name__ == '__main__':
    train()
        
