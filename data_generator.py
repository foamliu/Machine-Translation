# encoding=utf-8
import pickle

import numpy as np
from gensim.models import KeyedVectors
from keras.utils import Sequence

from config import batch_size, Tx, Ty, embedding_size, n_s
from config import start_word, start_embedding, unknown_word, unknown_embedding, stop_word, stop_embedding


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage

        print('loading CWV word embedding(zh)')
        self.word_vectors_zh = KeyedVectors.load_word2vec_format('data/sgns.merge.char')

        print('loading {} samples'.format(usage))
        if usage == 'train':
            samples_path = 'data/samples_train.p'
        else:
            samples_path = 'data/samples_valid.p'

        self.samples = pickle.load(open(samples_path, 'rb'))
        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(batch_size)))

    def __getitem__(self, idx):
        i = idx * batch_size

        length = min(batch_size, (len(self.samples) - i))

        batch_x = np.zeros((length, Tx, embedding_size), np.float32)
        batch_y = np.zeros((length, Ty), np.int32)

        s0 = np.zeros((length, n_s))
        c0 = np.zeros((length, n_s))

        for i_batch in range(length):
            sample = self.samples[i + i_batch]

            input_size = min(Tx, len(sample['input']))
            for idx in range(input_size):
                word = sample['input'][idx]
                if word == start_word:
                    embedding = start_embedding
                elif word == stop_word:
                    embedding = stop_embedding
                elif word == unknown_word:
                    embedding = unknown_embedding
                else:
                    embedding = self.word_vectors_zh[word]
                batch_x[i_batch, idx] = embedding

            output_size = min(Ty, len(sample['output']))
            for idx in range(output_size):
                batch_y[i_batch, idx] = sample['output'][idx]

        targets = list(batch_y.swapaxes(0, 1))
        return [batch_x, s0, c0], targets

    def on_epoch_end(self):
        np.random.shuffle(self.samples)


def train_gen():
    return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('valid')
