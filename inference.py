import io
import json
import tensorflow as tf
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
                config['max_len_p'], config['num_head'], config['num_layer'], config['vocab_size_c'] + 1,
                config['vocab_size_p'] + 1))
ckpt = tf.train.Checkpoint(clf=clf)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints/finetune', max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)


def inference(molecule_smiles: str, protein_fasta: str) -> float:
    data_c = make_seq([molecule_smiles], config['max_len_c'], vocab_c, config['vocab_size_c'], config['word_len_c'])
    data_p = make_seq([protein_fasta], config['max_len_p'], vocab_p, config['vocab_size_p'], config['word_len_p'])
    pred = clf(data_c, data_p, False)
    return float(tf.sigmoid(pred))
