import numpy as np
import re

def tokenize(text):
    ignore_chars = r"[.,'!?_-]"
    text = re.sub(ignore_chars, " ", text)
    words = text.lower().split()
    return [w for w in words if w != '']

if __name__ == '__main__':
    assert tokenize('abc def') == ['abc', 'def']
    assert tokenize('abc-def.ghi') == ['abc', 'def', 'ghi']
    assert tokenize('A  B  ') == ['a', 'b']
