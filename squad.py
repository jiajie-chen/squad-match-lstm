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

hidden_size = 50

out_file = open('loss.txt', 'a')

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
    # return a list of (input, output) tuples, where input is a tuple (passage, question) and output is an array (start, end) for the answer
    passage = vectorize(para.passage, fixed_length=passage_max_length)
    question = vectorize(qa.question, fixed_length=question_max_length)

    answers = []
    for answer in qa.answers:
        start = answer['answer_start']
        end = start + len(answer['text'])
        answers.append(np.array([start, end]))

    return [((passage, question), answer) for answer in answers]

def questions_from_dataset(ds):
    for para in ds.paragraphs:
        for qa in para.qas:
            yield (para, qa)

class Squad(Net):
    def setup(self):
        passage = tf.placeholder(tf.int32, [None, passage_max_length], name='passage')  # shape (batch_size, passage_max_length)
        question = tf.placeholder(tf.int32, [None, question_max_length], name='question')  # shape (batch_size, question_max_length)
        desired_output = tf.placeholder(tf.int32, [None, 2], name='desired_output')  # shape (batch_size, passage_max_length)
        
        embedding = tf.constant(embedding_matrix, name='embedding', dtype=tf.float32)


        #######################
        # Preprocessing layer #
        ####################### 

        passage_embedded = tf.nn.embedding_lookup(embedding, passage)  # shape (batch_size, passage_max_length, embedding_size)
        question_embedded = tf.nn.embedding_lookup(embedding, question)  # shape (batch_size, question_max_length, embedding_size)

        dropout = tf.placeholder(tf.float32)

        with tf.variable_scope('passage_lstm'):
            passage_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            passage_cell = tf.nn.rnn_cell.DropoutWrapper(passage_cell, output_keep_prob=dropout)
            passage_cell = tf.nn.rnn_cell.MultiRNNCell([passage_cell] * 2)
            H_p, _ = tf.nn.dynamic_rnn(passage_cell, passage_embedded, dtype=tf.float32)  # shape (batch_size, passage_max_length, hidden_size)
        
        with tf.variable_scope('question_lstm'):
            question_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            question_cell = tf.nn.rnn_cell.DropoutWrapper(question_cell, output_keep_prob=dropout)
            question_cell = tf.nn.rnn_cell.MultiRNNCell([question_cell] * 2)
            H_q, _ = tf.nn.dynamic_rnn(question_cell, question_embedded, dtype=tf.float32)  # shape (batch_size, question_max_length, hidden_size)

        H_p = tf.Print(H_p, [H_p], "H_p")

        ####################
        # Match-LSTM layer #
        ####################

        # Weights and bias to compute `G`
        W_q = self.weight_variable(shape=[hidden_size, hidden_size])
        W_p = self.weight_variable(shape=[hidden_size, hidden_size])
        W_r = self.weight_variable(shape=[hidden_size, hidden_size])
        b_p = self.bias_variable(shape=[hidden_size, 1]) # needs to be 50x1

        # Weight and bias to compute `a`
        w = self.weight_variable(shape=[hidden_size, 1])
        b_alpha = self.bias_variable(shape=[1, 1])   # In the paper, this is `b` (scalar value)

        # BATCHING STARTS HERE

        # Only calculate `WH_q` once
        WH_q = tf.matmul(W_q, H_q[0], transpose_b=True)

        WH_q = tf.Print(WH_q, [WH_q], "WH_q")

        # Results for forward and backward LSTMs
        H_r_forward = []
        H_r_backward = []

        with tf.variable_scope('forward_match_lstm') as scope:
            forward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True), output_keep_prob=dropout)
            forward_state = forward_cell.zero_state(1, dtype=tf.float32) #batch_size is 1??
            h = forward_state.h
            for i in range(passage_max_length): # len(H_p) = passage_max_length
                if i > 0: # lets you reuse h and forward_state per iteration
                    scope.reuse_variables()

                WH_p = tf.matmul(W_p, tf.reshape(H_p[0][i], [-1, 1])) # SQUARE PEG ROUND HOLE
                Wh_r = tf.matmul(W_r, h, transpose_b=True)

                G_forward = tf.tanh(WH_q + tf.tile((WH_p + Wh_r + b_p), [1, question_max_length]))
                wG_forward = tf.matmul(w, G_forward, transpose_a=True)
                alpha_forward = tf.nn.softmax(tf.transpose(wG_forward) + tf.tile(b_alpha, [question_max_length, 1]))

                z_forward = tf.concat(0, [tf.reshape(H_p[0][i], [-1, 1]), tf.matmul(H_q[0], alpha_forward, transpose_a=True)])
                z_forward = tf.transpose(z_forward)
                h, forward_state = forward_cell(z_forward, forward_state)
                H_r_forward.append(tf.transpose(h))

        with tf.variable_scope('backward_match_lstm') as scope:
            backward_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True), output_keep_prob=dropout)
            backward_state = backward_cell.zero_state(1, dtype=tf.float32) #batch_size is 1??
            h = backward_state.h
            for i in reversed(range(passage_max_length)):
                if i < passage_max_length-1: # lets you reuse h and backward_state per iteration
                    scope.reuse_variables()

                WH_p = tf.matmul(W_p, tf.reshape(H_p[0][i], [-1, 1]))
                Wh_r = tf.matmul(W_r, h, transpose_b=True)

                G_backward = tf.tanh(WH_q + tf.tile((WH_p + Wh_r + b_p), [1, question_max_length]))
                wG_backward = tf.matmul(w, G_backward, transpose_a=True)
                alpha_backward = tf.nn.softmax(tf.transpose(wG_backward) + tf.tile(b_alpha, [question_max_length, 1]))

                z_backward = tf.concat(0, [tf.reshape(H_p[0][i], [-1, 1]), tf.matmul(H_q[0], alpha_backward, transpose_a=True)])
                z_backward = tf.transpose(z_backward)
                h, backward_state = backward_cell(z_backward, backward_state)
                H_r_backward.append(tf.transpose(h))

        # After finding forward and backward `H_r[i]` for all `i`, concatenate `H_r_forward` and `H_r_backward`
        H_r_forward = tf.concat(1, H_r_forward)
        H_r_backward = tf.concat(1, H_r_backward)
        H_r = tf.concat(0, [H_r_forward, H_r_backward])

        H_r = tf.Print(H_r, [H_r], 'H_r')

        # TODO: Assert that the shape of `H_r` is (2 * hidden_size, passage_max_length)


        ########################
        # Answer-Pointer layer #
        ########################

        # Weights and bias to compute `F`
        V = self.weight_variable(shape=[hidden_size, 2 * hidden_size])
        W_a = self.weight_variable(shape=[hidden_size, hidden_size])
        b_a = self.bias_variable(shape=[hidden_size, 1])   # In the paper, this is `c`

        # Weight and bias to compute `beta`
        v = self.weight_variable(shape=[hidden_size, 1])
        b_beta = self.bias_variable(shape=[1, 1])

        # Only calculate `VH` once
        VH = tf.matmul(V, H_r)        # shape (hidden_size, passage_max_length)

        output = []

        with tf.variable_scope('answer_pointer_lstm') as scope:
            pointer_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True), output_keep_prob=dropout)
            pointer_state = pointer_cell.zero_state(1, dtype=tf.float32)
            h = pointer_state.h
            for k in range(2):
                if k > 0:
                    scope.reuse_variables()

                Wh_a = tf.matmul(W_a, h, transpose_b=True)

                F = tf.tanh(VH + tf.tile((Wh_a + b_a), [1, passage_max_length]))
                output_k = tf.matmul(v, F, transpose_a=True) + tf.tile(b_beta, [1, passage_max_length])
                output_k = tf.Print(output_k, [output_k], 'output_k')
                output.append(output_k)
                beta = tf.nn.softmax(output_k)

                h, pointer_state = pointer_cell(tf.transpose(tf.matmul(H_r, beta, transpose_b=True)), pointer_state)
        
        output = tf.concat(0, output)
        
        output = tf.Print(output, [output], "output")
        desired_output = tf.Print(desired_output, [desired_output], "desired_output")

        soft = tf.nn.sparse_softmax_cross_entropy_with_logits(output, desired_output[0])
        soft = tf.Print(soft, [soft], "soft")

        loss = tf.reduce_mean(soft)
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
        vectors = [v for vector in vectors for v in vector] # flatten vectors

        questions = np.array([question for ((_, question), _) in vectors])
        passages = np.array([passage for ((passage, _), _) in vectors])
        answers = np.array([answer for ((_, _), answer) in vectors])
        
        print "Passages: {0}   Questions: {1}   Answers: {2}".format(len(passages), len(questions), len(answers))
        feed = {self.passage: passages, self.question: questions, self.desired_output: answers, self.dropout: 0.5}
        _, loss = self.session.run([self.train_step, self.loss], feed_dict=feed)
        print "Current loss: {0}".format(loss)
        with open('loss.txt', 'a+') as f:
            f.write("{0}\n".format(loss))

    @staticmethod
    def weight_variable(shape, name=None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name=None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

def iterate_batches(list, size=10):
    i = 0
    while True:
        yield [list[i+j] for j in range(size)]
        i += size
        if i + j > len(list):
            i = 0

n = Squad(dir_path='save/squad3')

def train():
    questions = list(questions_from_dataset(dataset.train()))
    random.shuffle(questions)
    print "{0} Questions".format(len(questions))
    questions = [q for q in questions if len(q[0].passage) < 200]
    print "{0} Questions".format(len(questions))
    test_questions = list(questions_from_dataset(dataset.test()))
    random.shuffle(test_questions)
    
    i = 0
    for i, batch in enumerate(iterate_batches(questions, size=1)):
        n.train(batch)
        if i % 500 == 0:
            n.save(i)

def generate_heatmap(net, para, question):
    vectors = [vectors_from_question(p, q) for p, q in [(para, question)]]
    vectors = [v for vector in vectors for v in vector] # flatten vectors

    questions = np.array([q for ((p, q), a) in vectors])
    passages = np.array([p for ((p, q), a) in vectors])
    answer = net.session.run(net.output, {net.dropout: 1, net.question: questions, net.passage: passages})[0]
    
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
        
