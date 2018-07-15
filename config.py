import os
import numpy as np

batch_size = 160
epochs = 10000
patience = 50
num_train_samples = 9903244
num_valid_samples = 8000
embedding_size = 300
vocab_size_en = 50000
vocab_size_zh = 0
max_token_length_en = Tx = 50
max_token_length_zh = Ty = 50
# hidden state size of the post-attention LSTM
n_s = 256
# hidden state size of the Bi-LSTM
n_a = 128


train_folder = 'data/ai_challenger_translation_train_20170912'
valid_folder = 'data/ai_challenger_translation_validation_20170912'
test_a_folder = 'data/ai_challenger_translation_test_a_20170923'
test_b_folder = 'data/ai_challenger_translation_test_b_20171128'
train_translation_folder = os.path.join(train_folder, 'translation_train_20170912')
valid_translation_folder = os.path.join(valid_folder, 'translation_validation_20170912')
train_translation_en_filename = 'train.en'
train_translation_zh_filename = 'train.zh'
valid_translation_en_filename = 'valid.en'
valid_translation_zh_filename = 'valid.zh'

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<UNK>'
start_embedding = np.zeros((embedding_size,))
stop_embedding = np.ones((embedding_size,))
unknown_embedding = np.ones((embedding_size,)) / 2
