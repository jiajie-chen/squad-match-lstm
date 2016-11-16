# Setup

```
virtualenv venv
pip install --upgrade pip
pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc2-py2-none-any.whl
```

Look [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md) for your tensorflow URL.

- unzip the glove (word embeddings) file in data/ (it's gitignored b/c it's big)

- generate the .npy file that contains the embedding matrix (this is faster than loading it every time, but the file is too big for github)

```
mkdir cache
python embedding.py
```

# Every time you develop

```
source venv/bin/activate
```

# Open questions

How do we deal with the fact that many of the phrases/entities in the documents (e.g. "Denver Broncos") might not have meaningful embeddings?

- idea: in addition to word embeddings, augment the vector representation of each word in the passage with a sparse vector N-long (where N is the length of the question), where there's a 1 if that word appears at the corresponding index in the question

