import json
from helpers import print_distribution, tokenize

class Dataset(object):
    def __init__(self, path):
        data = json.load(open(path))
        self.paragraphs = [Paragraph(p) for article in data['data'] for p in article['paragraphs']]

class Paragraph(object):
    def __init__(self, data):
        self.passage = data['context']
        self.qas = [QA(qa) for qa in data['qas']]

class QA(object):
    def __init__(self, data):
        self.question = data['question']
        self.answers = data['answers'] # array of dictionaries, with keys `answer_start` (a character index) and `text`

def test():
    return Dataset('data/dev-v1.1.json')

def train():
    return Dataset('data/train-v1.1.json')

if __name__ == '__main__':
    t = train()
    passage_length_distribution = [len(tokenize(p.passage)) for p in t.paragraphs]
    question_length_distribution = [len(tokenize(q.question)) for para in t.paragraphs for q in para.qas]
    print "Passage length distribution (tokens):", print_distribution(passage_length_distribution)
    print "Question length distribution (tokens):", print_distribution(question_length_distribution)
