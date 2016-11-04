# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import abc
import sqlite3
from collections import Counter

from nltk import ngrams

from thot_utils.libs import config
from thot_utils.libs.utils import split_string_to_words


class LanguageModelProviderInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_count(self, src_words):
        pass

    @abc.abstractmethod
    def get_all_counts(self):
        pass


class LanguageModelFileProvider(LanguageModelProviderInterface):
    def __init__(self, fd, ngrams_length):
        self.fd = fd
        self.ngrams_length = ngrams_length
        self.main_counter = Counter()
        self.run()

    def run(self):
        for line in self.fd:
            word_array = split_string_to_words(line)
            self.train_word_array(word_array)

    def train_word_array(self, word_array):
        # obtain counts for 0-grams
        self.main_counter.update({"": len(word_array)})

        # obtain counts for higher order n-grams
        for i in range(1, self.ngrams_length + 1):
            self.main_counter.update(
                ngrams(word_array, i, pad_left=True, pad_right=True, left_pad_symbol=config.bos_str,
                       right_pad_symbol=config.eos_str)
            )

    def get_count(self, word):
        return self.main_counter[word]

    def get_all_counts(self):
        for source, count in self.main_counter.items():
            yield source, count


class LanguageModelDBProvider(LanguageModelProviderInterface):
    def __init__(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

    def get_count(self, word):
        self.cursor.execute('select c from ngram_counts where n=? limit 1', [word])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_all_counts(self):
        raise NotImplemented()

    def load_from_other_provider(self, provider):
        self.connection.execute('CREATE TABLE ngram_counts (n text primary key not null, c int not null)')
        for key, value in provider.get_all_counts():
            self.cursor.execute('insert into ngram_counts values (?, ?)', [' '.join(key), value])
        self.connection.commit()
