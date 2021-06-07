import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def make_pe(seq_len, d_model):
    pe = np.arange(seq_len)[:, np.newaxis] / np.power(10000, np.arange(d_model)[np.newaxis, :] // 2 * 2 / d_model)
    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])
    return tf.cast(pe, tf.float32)


def make_look_ahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)


def make_padding_mask(seq):
    return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]


def self_attention(q, k, v, mask):
    attn_weight = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(k.shape[-1], tf.float32))
    if mask is not None:
        attn_weight += mask * -1e9
    return tf.matmul(tf.nn.softmax(attn_weight, axis=-1), v)


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.depth = d_model // num_head
        self.dense_q = layers.Dense(d_model)
        self.dense_k = layers.Dense(d_model)
        self.dense_v = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split(self, x):
        return tf.transpose(tf.reshape(x, (-1, x.shape[1], self.num_head, self.depth)), perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask, training):
        q = self.split(self.dense_q(q, training=training))
        k = self.split(self.dense_k(k, training=training))
        v = self.split(self.dense_v(v, training=training))
        attn_out = self_attention(q, k, v, mask)
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        attn_out = tf.reshape(attn_out, (-1, attn_out.shape[1], self.d_model))
        return self.dense(attn_out, training=training)


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_head)
        self.ffn = layers.Dense(dff, activation='relu')
        self.dense = layers.Dense(d_model)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.ln1 = layers.LayerNormalization(epsilon=ln_epsilon)
        self.ln2 = layers.LayerNormalization(epsilon=ln_epsilon)

    def call(self, x, mask, training):
        mha_out = self.mha(x, x, x, mask, training)
        dropout1_out = self.dropout1(mha_out, training=training)
        ln1_out = self.ln1(dropout1_out + x, training=training)
        ffn_out = self.ffn(ln1_out, training=training)
        dense_out = self.dense(ffn_out, training=training)
        dropout2_out = self.dropout2(dense_out, training=training)
        ln2_out = self.ln2(dropout2_out + ln1_out, training=training)
        return ln2_out


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_head)
        self.mha2 = MultiHeadAttention(d_model, num_head)
        self.ffn = layers.Dense(dff, activation='relu')
        self.dense = layers.Dense(d_model)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.ln1 = layers.LayerNormalization(epsilon=ln_epsilon)
        self.ln2 = layers.LayerNormalization(epsilon=ln_epsilon)
        self.ln3 = layers.LayerNormalization(epsilon=ln_epsilon)

    def call(self, x, enc_out, look_ahead_mask, padding_mask, training):
        mha1_out = self.mha1(x, x, x, look_ahead_mask, training)
        dropout1_out = self.dropout1(mha1_out, training=training)
        ln1_out = self.ln1(dropout1_out + x, training=training)
        mha2_out = self.mha2(ln1_out, enc_out, enc_out, padding_mask, training)
        dropout2_out = self.dropout2(mha2_out, training=training)
        ln2_out = self.ln2(dropout2_out + ln1_out, training=training)
        ffn_out = self.ffn(ln2_out, training=training)
        dense_out = self.dense(ffn_out, training=training)
        dropout3_out = self.dropout3(dense_out, training=training)
        ln3_out = self.ln3(dropout3_out + ln2_out, training=training)
        return ln3_out


class Encoder(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head, num_layer, pe, vocab_size):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layer = num_layer
        self.pe = pe
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, dff, dropout, ln_epsilon, num_head) for _ in range(num_layer)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, mask, training):
        x = self.embedding(x, training=training) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pe[:x.shape[1], :]
        x = self.dropout(x, training=training)
        for k in range(self.num_layer):
            x = self.enc_layers[k](x, mask, training)
        return x


class Decoder(layers.Layer):
    def __init__(self, d_model, dff, dropout, ln_epsilon, num_head, num_layer, pe, vocab_size):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layer = num_layer
        self.pe = pe
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.dec_layers = [DecoderLayer(d_model, dff, dropout, ln_epsilon, num_head) for _ in range(num_layer)]
        self.dropout = layers.Dropout(dropout)

    def call(self, x, enc_out, look_ahead_mask, padding_mask, training):
        x = self.embedding(x, training=training) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pe[:x.shape[1], :]
        x = self.dropout(x, training=training)
        for k in range(self.num_layer):
            x = self.dec_layers[k](x, enc_out, look_ahead_mask, padding_mask, training)
        return x


class Transformer(tf.keras.Model):
    def __init__(self, d_model, dff, dropout, ln_epsilon, max_seq_len_enc, max_seq_len_dec, num_head, num_layer,
                 vocab_size_enc, vocab_size_dec):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, dff, dropout, ln_epsilon, num_head, num_layer,
                               make_pe(max_seq_len_enc, d_model), vocab_size_enc)
        self.decoder = Decoder(d_model, dff, dropout, ln_epsilon, num_head, num_layer,
                               make_pe(max_seq_len_dec, d_model), vocab_size_dec)
        self.dense = layers.Dense(vocab_size_dec)

    def call(self, x_enc, x_dec, training):
        mask = make_padding_mask(x_enc)
        combined_mask = tf.maximum(make_look_ahead_mask(x_dec.shape[1]), make_padding_mask(x_dec))
        enc_out = self.encoder(x_enc, mask, training)
        dec_out = self.decoder(x_dec, enc_out, combined_mask, mask, training)
        return self.dense(dec_out, training=training)


class Classifier(tf.keras.Model):
    def __init__(self, tfm):
        super(Classifier, self).__init__()
        self.tfm = tfm
        self.dense = layers.Dense(1)

    def call(self, x_enc, x_dec, training):
        tfm_out = self.tfm(x_enc, x_dec, training)
        dense_out = self.dense(tfm_out[:, -1, :], training=training)
        return tf.reshape(dense_out, -1)
