import tensorflow as tf
from transformer import Transformer

tmp = Transformer(512, 2048, 0.1, 1e-6, 100, 200, 8, 4, 8500, 8000)
tmp_in_enc = tf.random.uniform((64, 43))
tmp_in_dec = tf.random.uniform((64, 56))
tmp_out = tmp(tmp_in_enc, tmp_in_dec, False)
print(tmp_out.shape)
