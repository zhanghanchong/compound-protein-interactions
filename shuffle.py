import io

import numpy as np


def shuffle(filenames):
    for filename in filenames:
        cpis = io.open('./datasets/raw/' + filename).read().split('\n')
        np.random.shuffle(cpis)
        size = len(cpis)
        file = io.open('./datasets/shuffle/' + filename, 'w')
        for i in range(size):
            file.write(cpis[i])
            if i < size - 1:
                file.write('\n')
        file.close()
