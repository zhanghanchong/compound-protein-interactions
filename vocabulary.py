import io
import json

from tensorflow.keras import preprocessing


def vocabulary(filenames):
    tokenizer_c = preprocessing.text.Tokenizer(filters='')
    tokenizer_p = preprocessing.text.Tokenizer(filters='')
    for filename in filenames:
        cpis = io.open('./datasets/raw/' + filename).read().split('\n')
        tokenizer_c.fit_on_texts(['<start> ' + ' '.join(cpi.split(' ')[0]) + ' <end>' for cpi in cpis])
        tokenizer_p.fit_on_texts(['<start> ' + ' '.join(cpi.split(' ')[1]) + ' <end>' for cpi in cpis])
    io.open('./vocabularies/compound.json', 'w').write(json.dumps(tokenizer_c.word_index, indent=4))
    io.open('./vocabularies/protein.json', 'w').write(json.dumps(tokenizer_p.word_index, indent=4))
