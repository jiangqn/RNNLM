import os
import yaml
import numpy as np
from src.data_process.utils import load_raw_data, build_vocab, save_data, load_glove
from src.utils.logger import Logger

def preprocess(args):
    base_path = os.path.join('./data', args.data)
    raw_base_path = os.path.join(base_path, 'raw')
    raw_train_path = os.path.join(raw_base_path, 'train.txt')
    raw_valid_path = os.path.join(raw_base_path, 'valid.txt')
    raw_test_path = os.path.join(raw_base_path, 'test.txt')
    processed_base_path = os.path.join(base_path, 'processed')
    processed_train_path = os.path.join(processed_base_path, 'train.npz')
    processed_valid_path = os.path.join(processed_base_path, 'valid.npz')
    processed_test_path = os.path.join(processed_base_path, 'test.npz')
    glove_path = os.path.join(processed_base_path, 'glove.npy')
    log_base_path = os.path.join(base_path, 'log')
    log_path = os.path.join(log_base_path, 'preprocess_log.txt')
    data_log_path = os.path.join(log_base_path, 'data_log.yml')
    if not os.path.exists(processed_base_path):
        os.makedirs(processed_base_path)
    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path)
    logger = Logger(log_path)
    logger.log('preprocess start')
    logger.log('load train data')
    train_data = load_raw_data(raw_train_path)
    logger.log('load valid data')
    valid_data = load_raw_data(raw_valid_path)
    logger.log('load test data')
    test_data = load_raw_data(raw_test_path)
    logger.log('build vocab')
    word2index, index2word = build_vocab(train_data)
    total_words = len(word2index)
    vocab_size = len(index2word)
    oov_words = total_words - vocab_size
    logger.log('total_words: %d' % total_words)
    logger.log('vocab_size: %d' % vocab_size)
    logger.log('oov_words: %d' % oov_words)
    logger.log('save train data')
    save_data(train_data, processed_train_path, word2index)
    logger.log('save valid data')
    save_data(valid_data, processed_valid_path, word2index)
    logger.log('save test data')
    save_data(test_data, processed_test_path, word2index)
    logger.log('load glove')
    glove = load_glove(args.glove_source_path, vocab_size, word2index)
    np.save(glove_path, glove)
    data_log = {
        'vocab_size': vocab_size,
        'total_words': total_words,
        'oov_words': oov_words
    }
    logger.log('save data_log')
    with open(data_log_path, 'w') as handle:
        yaml.safe_dump(data_log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)
    logger.log('preprocess finish')