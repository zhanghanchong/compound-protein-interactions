import io
import json
import tensorflow as tf
from optimizer import make_optimizer
from preprocess import make_seq
from transformer import Classifier
from transformer import Transformer

with io.open('./parameters.json') as file:
    config = json.load(file)
with io.open('./vocabularies/compound.json') as file:
    vocab_c = json.load(file)
with io.open('./vocabularies/protein.json') as file:
    vocab_p = json.load(file)
clf = Classifier(
    Transformer(config['d_model'], config['dff'], config['dropout'], config['ln_epsilon'], config['max_len_c'],
                config['max_len_p'], config['num_head'], config['num_layer'], len(vocab_c) + 1, len(vocab_p) + 1))
opt = make_optimizer(config['adam_beta_1'], config['adam_beta_2'], config['adam_epsilon'], config['d_model'],
                     config['warmup'])
ckpt = tf.train.Checkpoint(clf=clf, opt=opt)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints/finetune', max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)


def inference(molecule_smiles: str, protein_fasta: str) -> float:
    data_c = make_seq([molecule_smiles], vocab_c)
    data_p = make_seq([protein_fasta], vocab_p)
    pred = clf(data_c, data_p, False)
    return float(tf.sigmoid(pred))


while 1:
    cp = input()
    pt = input()
    print(inference(cp, pt))
