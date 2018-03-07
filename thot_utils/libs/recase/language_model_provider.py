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
    def __init__(self, raw_fd, ngrams_length):
        self.fd = raw_fd
        self.ngrams_length = ngrams_length

    def train_word_array(self, counter, word_array):
        # obtain counts for 0-grams
        counter[""] += len(word_array)

        # obtain counts for higher order n-grams
        for i in range(1, self.ngrams_length + 1):
            keys = ngrams(
                word_array, i, pad_left=True, pad_right=True, left_pad_symbol=config.bos_str,
                right_pad_symbol=config.eos_str
            )
            for key in keys:
                counter[' '.join(key)] += 1

    def generate_sqlite(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()
        self.connection.execute('PRAGMA synchronous=OFF')
        self.connection.execute('PRAGMA cache_size=-2000000')

        self.connection.execute('DROP TABLE IF EXISTS ngram_counts')
        self.connection.execute('CREATE TABLE ngram_counts (n TEXT PRIMARY KEY NOT NULL, c INT NOT NULL)')
        counter = Counter()
        for idx, line in enumerate(self.fd):
            word_array = split_string_to_words(line)
            self.train_word_array(counter, word_array)
            if idx % 100000 == 0:
                print(idx)
                self.update_ngram_counts(counter)
                # self.connection.commit()
                counter = Counter()
        if counter:
            self.update_ngram_counts(counter)

        self.connection.commit()

    def update_ngram_counts(self, counter):
        items = counter.items()
        keys = [(i[0], ) for i in items]
        self.cursor.executemany('insert or ignore into ngram_counts values (?1, 0)', keys)
        self.cursor.executemany('update ngram_counts set c=c+?2 where n=?1', items)


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
