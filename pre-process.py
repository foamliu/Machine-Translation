import os
import pickle
import xml.etree.ElementTree
import zipfile
from collections import Counter

import jieba
import nltk
from gensim.models import KeyedVectors
from tqdm import tqdm

from config import start_word, stop_word, unknown_word, vocab_size_en
from config import train_folder, valid_folder, test_a_folder, test_b_folder
from config import train_translation_folder, train_translation_zh_filename, train_translation_en_filename
from config import valid_translation_folder, valid_translation_zh_filename, valid_translation_en_filename
from utils import ensure_folder


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


def build_train_vocab_en():
    print('loading en word embedding')
    translation_path = os.path.join(train_translation_folder, train_translation_en_filename)

    with open(translation_path, 'r') as f:
        data = f.readlines()

    vocab = []
    max_len = 0
    longest_sentence = None
    print('scanning train data (en)')
    for sentence in tqdm(data):
        tokens = nltk.word_tokenize(sentence.strip().lower())
        for token in tokens:
            vocab.append(token)
        length = len(tokens)
        if length > max_len:
            longest_sentence = '/'.join(tokens)
            max_len = length

    counter = Counter(vocab)
    common = counter.most_common(vocab_size_en - 3)
    covered_count = 0
    for item in tqdm(common):
        covered_count += item[1]

    vocab = [item[0] for item in common]
    vocab.append(start_word)
    vocab.append(stop_word)
    vocab.append(unknown_word)

    print('max_len(zh): ' + str(max_len))
    print('longest_sentence: ' + longest_sentence)
    print('count of words in text (zh): ' + str(len(list(counter.keys()))))
    print('vocab size (zh): ' + str(len(vocab)))
    total_count = len(list(counter.elements()))
    print('coverage: ' + str(covered_count / total_count))

    filename = 'data/vocab_train_zh.p'
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def build_train_vocab_zh():
    print('loading word embedding(zh)')
    word_vectors = KeyedVectors.load_word2vec_format('data/wiki.en.vec')
    translation_path = os.path.join(train_translation_folder, train_translation_zh_filename)

    with open(translation_path, 'r') as f:
        data = f.readlines()

    vocab = []
    max_len = 0
    longest_sentence = None
    print('building {} train vocab (zh)')
    for sentence in tqdm(data):
        seg_list = jieba.cut(sentence.strip().lower())

        length = 0
        for word in seg_list:
            vocab.append(word)
            length = length + 1

        if length > max_len:
            longest_sentence = '/'.join(seg_list)
            max_len = length

    counter = Counter(vocab)
    total_count = 0
    covered_count = 0
    for word in tqdm(counter.keys()):
        total_count += counter[word]
        try:
            v = word_vectors[word]
            covered_count += counter[word]
        except (NameError, KeyError):
            # print(word)
            pass

    vocab = list(word_vectors.vocab.keys())
    vocab.append(start_word)
    vocab.append(stop_word)
    vocab.append(unknown_word)
    vocab = sorted(vocab)

    print('max_len(zh): ' + str(max_len))
    print('count of words in text (zh): ' + str(len(list(counter.keys()))))
    print('CWV vocab size (zh): ' + str(len(vocab)))
    print('coverage: ' + str(covered_count / total_count))
    print('longest_sentence: ' + longest_sentence)

    filename = 'data/vocab_train_zh.p'
    with open(filename, 'wb') as file:
        pickle.dump(vocab, file)


def extract_valid_data():
    valid_translation_path = os.path.join(valid_translation_folder, 'valid.en-zh.en.sgm')
    with open(valid_translation_path, 'r') as f:
        data_en = f.readlines()
    data_en = [line.replace(' & ', ' &amp; ') for line in data_en]
    with open(valid_translation_path, 'w') as f:
        f.writelines(data_en)

    root = xml.etree.ElementTree.parse(valid_translation_path).getroot()
    data_en = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.en'), 'w') as out_file:
        out_file.writelines(data_en)

    root = xml.etree.ElementTree.parse(os.path.join(valid_translation_folder, 'valid.en-zh.zh.sgm')).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_translation_folder, 'valid.zh'), 'w') as out_file:
        out_file.writelines(data_zh)


def build_samples():
    vocab_zh = pickle.load(open('data/vocab_train_zh.p', 'rb'))
    vocab_set_zh = set(vocab_zh)

    vocab_en = pickle.load(open('data/vocab_train_en.p', 'rb'))
    idx2word_en = vocab_en
    word2idx_en = dict(zip(idx2word_en, range(len(vocab_en))))

    for usage in ['train', 'valid']:
        if usage == 'train':
            translation_path_en = os.path.join(train_translation_folder, train_translation_en_filename)
            translation_path_zh = os.path.join(train_translation_folder, train_translation_zh_filename)
            filename = 'data/samples_train.p'
        else:
            translation_path_en = os.path.join(valid_translation_folder, valid_translation_en_filename)
            translation_path_zh = os.path.join(valid_translation_folder, valid_translation_zh_filename)
            filename = 'data/samples_valid.p'

        print('loading {} texts and vocab'.format(usage))
        with open(translation_path_en, 'r') as f:
            data_en = f.readlines()

        with open(translation_path_zh, 'r') as f:
            data_zh = f.readlines()

        print('building {} samples'.format(usage))
        samples = []
        for idx in tqdm(range(len(data_en))):
            sentence_zh = data_zh[idx].strip().lower()
            input_zh = []
            seg_list = jieba.cut(sentence_zh)

            for token in seg_list:
                if token in vocab_set_zh:
                    word = token
                else:
                    word = unknown_word
                input_zh.append(word)
            input_zh.append(stop_word)

            sentence_en = data_en[idx].strip().lower()
            tokens = nltk.word_tokenize(sentence_en)
            output_en = []
            for j, token in enumerate(tokens):
                try:
                    idx = word2idx_en[token]
                except (NameError, KeyError):
                    idx = word2idx_en[unknown_word]
                output_en.append(idx)
            output_en.append(word2idx_en[stop_word])

            samples.append({'input': list(input_zh), 'output': list(output_en)})
        with open(filename, 'wb') as f:
            pickle.dump(samples, f)
        print('{} {} samples created at: {}.'.format(len(samples), usage, filename))


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(train_folder):
        extract(train_folder)

    if not os.path.isdir(valid_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_folder):
        extract(test_a_folder)

    if not os.path.isdir(test_b_folder):
        extract(test_b_folder)

    if not os.path.isfile('data/vocab_train_zh.p'):
        build_train_vocab_zh()

    if not os.path.isfile('data/vocab_train_en.p'):
        build_train_vocab_en()

    extract_valid_data()

    if not os.path.isfile('data/samples_train.p') or not os.path.isfile('data/samples_valid.p'):
        build_samples()
