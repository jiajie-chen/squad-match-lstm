from net import Net, batch_generator
import tensorflow as tf
import numpy as np
from embedding import UNKNOWN, embeddings
from helpers import tokenize
import dataset
from util import rand_variable, weight_var
import random
import show_html
import sys

vocab, embedding_matrix = embeddings()
vocab_lookup = {word: i for i, word in enumerate(vocab)}
vocab_size, embedding_size = embedding_matrix.shape

passage_max_length = 200
question_max_length = 20

hp_size = 50
hq_size = 50
hr_size = 100

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
        
        def create_dense(input, input_size, output_size, relu=True):
            weights = weight_var([input_size, output_size])
            biases = weight_var([output_size])
            r = tf.matmul(input, weights) + biases
            return tf.nn.relu(r) if relu else r

        passage = tf.placeholder(tf.int32, [None, passage_max_length], name='passage')  # shape (batch_size, passage_max_length)
        question = tf.placeholder(tf.int32, [None, question_max_length], name='question')  # shape (batch_size, question_max_length)
        desired_output = tf.placeholder(tf.float32, [None, passage_max_length], name='desired_output')  # shape (batch_size, passage_max_length)
        
        embedding = tf.constant(embedding_matrix, name='embedding', dtype=tf.float32)

        #######################
        # Preprocessing layer #
        ####################### 

        passage_embedded = tf.nn.embedding_lookup(embedding, passage)  # shape (batch_size, passage_max_length, embedding_size)
        question_embedded = tf.nn.embedding_lookup(embedding, question)  # shape (batch_size, question_max_length, embedding_size)

        dropout = tf.placeholder(tf.float32)

        with tf.variable_scope('passage_lstm'):
            passage_cell = tf.nn.rnn_cell.LSTMCell(hp_size)
            passage_cell = tf.nn.rnn_cell.DropoutWrapper(passage_cell, output_keep_prob=dropout)
            passage_cell = tf.nn.rnn_cell.MultiRNNCell([passage_cell] * 2)
            hidden_p, _ = tf.nn.dynamic_rnn(passage_cell, passage_embedded, dtype=tf.float32)  # shape (batch_size, passage_max_length, hp_size)
        
        with tf.variable_scope('question_lstm'):
            question_cell = tf.nn.rnn_cell.LSTMCell(hq_size)
            question_cell = tf.nn.rnn_cell.DropoutWrapper(question_cell, output_keep_prob=dropout)
            question_cell = tf.nn.rnn_cell.MultiRNNCell([question_cell] * 2)
            hidden_q, _ = tf.nn.dynamic_rnn(question_cell, question_embedded, dtype=tf.float32)  # shape (batch_size, question_max_length, hq_size)

        ####################
        # Match-LSTM layer #
        ####################

        # TODO: this layer

        ########################
        # Answer-Pointer layer #
        ########################

        # TODO: this layer


        
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

n = Squad(dir_path='save/squad1')

def train():
    questions = list(questions_from_dataset(dataset.train()))
    random.shuffle(questions)
    test_questions = list(questions_from_dataset(dataset.test()))
    random.shuffle(test_questions)
    
    i = 0
    for i, batch in enumerate(iterate_batches(questions, size=20)):
        n.train(batch)
        if i % 10 == 0:
            n.save(i)

def generate_heatmap(net, para, question):
    vectors = [vectors_from_question(p, q) for p, q in [(para, question)]]
    questions = np.array([q for ((p, q), mask) in vectors])
    passages = np.array([p for ((p, q), mask) in vectors])
    mask = net.session.run(net.output, {net.dropout: 1, net.question: questions, net.passage: passages})[0]
    
    top_n = sorted(range(len(mask)), key=lambda i: mask[i], reverse=True)[:10]
    mask = [(1 if i in top_n else 0) for i in range(len(mask))]
    
    tokens = tokenize(para.passage)
    heatmap = u" ".join([u"<span style='background-color: rgba(255,0,0,{0})'>{1}</span>".format(max(0, min(1, value)), word) for value, word in zip(mask, tokens)])
    html = u"<h1>{0}</h1> <p>{1}</p>".format(question.question, heatmap)
    return html

def show_heatmap():
    test_questions = list(questions_from_dataset(dataset.test()))
    random.shuffle(test_questions)
    para, question = test_questions[0]
    show_html.show(generate_heatmap(n, para, question).encode('utf-8'))

if __name__ == '__main__':
    if 'show' in sys.argv:
        show_heatmap()
    else:
        train()
        
