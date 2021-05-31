import io
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import preprocessing


def preprocess(filenames):
    tokenizer_c = preprocessing.text.Tokenizer(filters='', lower=False)
    tokenizer_p = preprocessing.text.Tokenizer(filters='', lower=False)
    for filename in filenames:
        with io.open('./datasets/raw/' + filename) as file:
            cpis = file.read().split('\n')
        tokenizer_c.fit_on_texts(['<start> ' + ' '.join(cpi.split(' ')[0]) + ' <end>' for cpi in cpis])
        tokenizer_p.fit_on_texts(['<start> ' + ' '.join(cpi.split(' ')[1]) + ' <end>' for cpi in cpis])
        np.random.shuffle(cpis)
        with io.open('./datasets/shuffle/' + filename, 'w') as file:
            for i in range(len(cpis)):
                file.write(cpis[i])
                if i < len(cpis) - 1:
                    file.write('\n')
    with io.open('./vocabularies/compound.json', 'w') as file:
        file.write(json.dumps(tokenizer_c.word_index, indent=4))
    with io.open('./vocabularies/protein.json', 'w') as file:
        file.write(json.dumps(tokenizer_p.word_index, indent=4))


def make_seq(texts, vocab):
    batch_size = len(texts)
    seq_len = 0
    for text in texts:
        if len(text) > seq_len:
            seq_len = len(text)
    seq = np.zeros((batch_size, seq_len + 2))
    for i in range(batch_size):
        seq[i][0] = vocab['<start>']
        for j in range(len(texts[i])):
            if texts[i][j] in vocab:
                seq[i][j + 1] = vocab[texts[i][j]]
        seq[i][len(texts[i]) + 1] = vocab['<end>']
    return tf.cast(seq, tf.float32)
