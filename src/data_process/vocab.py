import operator
from src.utils.constants import PAD, UNK, SOS, EOS

class Vocab(object):

    def __init__(self):
        self.count_dict = dict()
        self.predefined_list = [PAD, UNK, SOS, EOS]

    def add(self, word):
        if word in self.count_dict:
            self.count_dict[word] += 1
        else:
            self.count_dict[word] = 1

    def add_list(self, words):
        for word in words:
            self.add(word)

    def get_vocab(self, max_size=None, min_freq=0):
        sorted_words = sorted(self.count_dict.items(), key=operator.itemgetter(1), reverse=True)
        word2index = {}
        for word in self.predefined_list:
            word2index[word] = len(word2index)
        for word, freq in sorted_words:
            if word in word2index:
                continue
            if (max_size is not None and len(word2index) >= max_size) or freq < min_freq:
                word2index[word] = word2index[UNK]
            else:
                word2index[word] = len(word2index)
        index2word = {}
        index2word[word2index[UNK]] = UNK
        for word, index in word2index.items():
            if index == word2index[UNK]:
                continue
            else:
                index2word[index] = word
        return word2index, index2word