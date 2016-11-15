# Setup

```
virtualenv venv
pip install --upgrade pip
pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.11.0rc2-py2-none-any.whl
```

Look [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md) for your tensorflow URL.

- unzip the glove (word embeddings) file in data/ (it's gitignored b/c it's big)

# Every time you develop

```
source venv/bin/activate
```
