import io
import json
import tensorflow as tf
from optimizer import accuracy_function
from optimizer import loss_function
from optimizer import make_optimizer
from tensorflow.keras import layers
from transformer import Transformer

with io.open('./config.json') as file:
    config = json.load(file)
with io.open('./vocabularies/compound.json') as file:
    vocab_c = json.load(file)
with io.open('./vocabularies/protein.json') as file:
    vocab_p = json.load(file)


class Classifier(tf.keras.Model):
    def __init__(self, tfm):
        super(Classifier, self).__init__()
        self.tfm = tfm
        self.dense = layers.Dense(1)

    def call(self, x_enc, x_dec, training):
        tfm_out = self.tfm(x_enc, x_dec, training)
        dense_out = self.dense(tfm_out[:, -1, :], training=training)
        return tf.reshape(dense_out, -1)


classifier = Classifier(
    Transformer(config['d_model'], config['dff'], config['dropout'], config['ln_epsilon'], config['max_len_c'],
                config['max_len_p'], config['num_head'], config['num_layer'], len(vocab_c) + 1, len(vocab_p) + 1))
optimizer = make_optimizer(config['adam_beta_1'], config['adam_beta_2'], config['adam_epsilon'], config['d_model'],
                           config['warmup'])
