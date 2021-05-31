import io
import json
import numpy as np
import tensorflow as tf
from optimizer import accuracy_function
from optimizer import loss_function
from optimizer import make_optimizer
from preprocess import make_seq
from tensorflow.keras import layers
from tensorflow.keras import metrics
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


clf = Classifier(
    Transformer(config['d_model'], config['dff'], config['dropout'], config['ln_epsilon'], config['max_len_c'],
                config['max_len_p'], config['num_head'], config['num_layer'], len(vocab_c) + 1, len(vocab_p) + 1))
opt = make_optimizer(config['adam_beta_1'], config['adam_beta_2'], config['adam_epsilon'], config['d_model'],
                     config['warmup'])
ckpt = tf.train.Checkpoint(clf=clf, opt=opt)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=5)
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
                cps = []
                pts = []
                data_i = np.zeros(config['batch_size'])
                for j in range(config['batch_size']):
                    cpi = file_.readline()
                    if len(cpi) == 0:
                        break
                    cp, pt, itr = cpi.split(' ')
                    cps.append(cp)
                    pts.append(pt)
                    data_i[j] = int(itr)
                if len(cps) < config['batch_size']:
                    break
                data_c = make_seq(cps, vocab_c)
                data_p = make_seq(pts, vocab_p)
                with tf.GradientTape() as tape:
                    pred = clf(data_c, data_p, True)
                    loss = loss_function(data_i, pred)
                grad = tape.gradient(loss, clf.trainable_variables)
                opt.apply_gradients(zip(grad, clf.trainable_variables))
                train_loss(loss)
                train_accuracy(accuracy_function(data_i, pred))
                print(
                    f'Epoch {i + 1} Batch {batch_id} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
        if (i + 1) % 5 == 0:
            ckpt_manager.save()
        print(f'Epoch {i + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')


train('./datasets/shuffle/train.txt')


def evaluate(filename):
    cnt_correct = 0
    cnt_total = 0
    with io.open(filename) as file_:
        while 1:
            cpi = file_.readline()
            if len(cpi) == 0:
                break
            cp, pt, itr = cpi.split(' ')
            data_c = make_seq([cp], vocab_c)
            data_p = make_seq([pt], vocab_p)
            pred = clf(data_c, data_p, False)
            cnt_correct += accuracy_function(np.array([int(itr)]), pred)
            cnt_total += 1
    return float(cnt_correct / cnt_total)


print(f"Development Set Accuracy {evaluate('./datasets/shuffle/dev.txt'):.4f}")
print(f"Test Set Accuracy {evaluate('./datasets/shuffle/test.txt'):.4f}")
