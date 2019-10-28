import os
import numpy as np
from src.data_process.tokenizer import tokenize
from src.utils.constants import PAD_INDEX, UNK_INDEX, SOS_INDEX, EOS_INDEX
from src.data_process.vocab import Vocab

def load_raw_data(path):
    file = open(path, 'r', encoding='utf-8')
    data = []
    for line in file.readlines():
        line = tokenize(line.strip())
        data.append(line)
    return data

def build_vocab(data):
    vocab = Vocab()
    for tokens in data:
        vocab.add_list(tokens)
    word2index, index2word = vocab.get_vocab()
    return word2index, index2word

def save_data(data, save_path, word2index):
    f = lambda x: word2index[x] if x in word2index else UNK_INDEX
    g = lambda y: [f(x) for x in y]
    src = []
    trg = []
    max_len = 0
    for tokens in data:
        tokens = g(tokens)
        src.append([SOS_INDEX] + tokens)
        trg.append(tokens + [EOS_INDEX])
        max_len = max(max_len, len(tokens) + 1)
    num = len(src)
    for i in range(num):
        src[i].extend([PAD_INDEX] * (max_len - len(src[i])))
        trg[i].extend([PAD_INDEX] * (max_len - len(trg[i])))
    src = np.asarray(src, dtype=np.int32)
    trg = np.asarray(trg, dtype=np.int32)
    np.savez(save_path, src=src, trg=trg)

def load_glove(path, vocab_size, word2index):
    if not os.path.isfile(path):
        raise IOError('Not a file', path)
    glove = np.random.uniform(-0.01, 0.01, [vocab_size, 300])
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                glove[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    glove[PAD_INDEX, :] = 0
    return glove