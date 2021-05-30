import tensorflow as tf
from transformer import make_pe
from transformer import Encoder
from transformer import Decoder

tmp_enc = Encoder(512, 2048, 0.1, 1e-6, 8, 4, make_pe(43, 512), 8000)
tmp_in_enc = tf.random.uniform((64, 43))
tmp_out_enc = tmp_enc(tmp_in_enc, None, False)
tmp_dec = Decoder(512, 2048, 0.1, 1e-6, 8, 4, make_pe(56, 512), 8500)
tmp_in_dec = tf.random.uniform((64, 56))
tmp_out_dec = tmp_dec(tmp_in_dec, tmp_out_enc, None, None, False)
print(tmp_out_enc.shape)
print(tmp_out_dec.shape)
