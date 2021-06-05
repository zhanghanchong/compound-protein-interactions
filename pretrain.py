import io
import json
import numpy as np
import tensorflow as tf
from optimizer import accuracy_function_tfm
from optimizer import loss_function_tfm
from optimizer import make_optimizer
from preprocess import make_batch
from tensorflow.keras import metrics
from transformer import Transformer

with io.open('./parameters.json') as file:
    config = json.load(file)
with io.open('./vocabularies/compound.json') as file:
    vocab_c = json.load(file)
with io.open('./vocabularies/protein.json') as file:
    vocab_p = json.load(file)
tfm = Transformer(config['d_model'], config['dff'], config['dropout'], config['ln_epsilon'], config['max_len_c'],
                  config['max_len_p'], config['num_head'], config['num_layer'], len(vocab_c) + 1, len(vocab_p) + 1)
opt = make_optimizer(config['adam_beta_1'], config['adam_beta_2'], config['adam_epsilon'], config['d_model'],
                     config['warmup'])
ckpt = tf.train.Checkpoint(tfm=tfm, opt=opt)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints/pretrain', max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.Mean(name='train_accuracy')


def train(filename):
    for i in range(config['epoch']):
        train_loss.reset_states()
        train_accuracy.reset_states()
        batch_id = 0
        with io.open(filename) as file_:
            while 1:
                batch_id += 1
                data_c, data_p_in, _ = make_batch(file_, config['batch_size'], vocab_c, vocab_p)
                if len(data_c) < config['batch_size']:
                    break
                data_p_out = np.zeros(data_p_in.shape)
                data_p_out[:, :-1] = data_p_in[:, 1:]
                data_p_out = tf.cast(data_p_out, tf.int64)
                with tf.GradientTape() as tape:
                    pred = tfm(data_c, data_p_in, True)
                    loss = loss_function_tfm(data_p_out, pred)
                grad = tape.gradient(loss, tfm.trainable_variables)
                opt.apply_gradients(zip(grad, tfm.trainable_variables))
                train_loss(loss)
                train_accuracy(accuracy_function_tfm(data_p_out, pred))
                print(
                    f'Epoch {i + 1} Batch {batch_id} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        ckpt_manager.save()
        print(f'Epoch {i + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')


train('./datasets/shuffle/large_data.csv')


def evaluate(filename):
    cnt_correct = 0
    cnt_total = 0
    with io.open(filename) as file_:
        while 1:
            data_c, data_p_in, _ = make_batch(file_, 1, vocab_c, vocab_p)
            if len(data_c) < 1:
                break
            data_p_out = np.zeros(data_p_in.shape)
            data_p_out[:, :-1] = data_p_in[:, 1:]
            data_p_out = tf.cast(data_p_out, tf.int64)
            pred = tfm(data_c, data_p_in, False)
            cnt_correct += accuracy_function_tfm(data_p_out, pred)
            cnt_total += 1
    return float(cnt_correct / cnt_total)


print(f"Test Set Accuracy {evaluate('./datasets/shuffle/small_data.csv'):.4f}")
