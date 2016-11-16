import numpy as np
import re

def tokenize(text):
    ignore_chars = r"[.,'!?_-]"
    text = re.sub(ignore_chars, " ", text)
    words = text.lower().split()
    return [w for w in words if w != '']

def print_distribution(numbers):
    n = sorted(numbers)
    return "{0} - {1}, with median {2} and 90th-percentile {3}".format(n[0], n[-1], n[len(n)/2], n[int(len(n)*0.9)])

if __name__ == '__main__':
    assert tokenize('abc def') == ['abc', 'def']
    assert tokenize('abc-def.ghi') == ['abc', 'def', 'ghi']
    assert tokenize('A  B  ') == ['a', 'b']
