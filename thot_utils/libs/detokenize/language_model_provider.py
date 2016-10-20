# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import abc
from collections import Counter

import sqlite3

import sys
from nltk import ngrams
from thot_utils.libs import config
from thot_utils.libs.utils import transform_word, is_categ, split_string_to_words


class LanguageModelProviderInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_count(self, src_words):
        pass

    @abc.abstractmethod
    def get_all_counts(self):
        pass


class LanguageModelFileProvider(LanguageModelProviderInterface):
    def __init__(self, raw_fd, tokenized_fd, ngrams_length):
        self.raw_fd = raw_fd
        self.tokenized_fd = tokenized_fd
        self.ngrams_length = ngrams_length
        self.main_counter = Counter()
        self.run()

    def run(self):
        lmvoc = {}
        # Read parallel files line by line
        for rline, tline in zip(self.raw_fd, self.tokenized_fd):
            raw_word_array = split_string_to_words(rline)
            tok_array = split_string_to_words(tline)
            # Process sentence
            retval = self.train_sent_tok(raw_word_array, tok_array, lmvoc)
            if not retval:
                print("Warning: something went wrong while training the language model for sentence", file=sys.stderr)

    def train_sent_tok(self, raw_word_array, tok_array, lmvoc):
        if (len(tok_array) > 0):
            # train translation model for sentence
            i = 0
            j = 0
            prev_j = 0
            error = False

            # Obtain transformed raw word array
            trans_raw_word_array = []
            while (i < len(raw_word_array)):
                end = False
                str = ""

                # process current raw word
                while (end == False):
                    if (raw_word_array[i] == str):
                        end = True
                    else:
                        if (j >= len(tok_array)):
                            error = True
                            end = True
                        else:
                            str = str + tok_array[j]
                            j = j + 1

                # Check that no errors were found while processing current raw word
                if (error == True):
                    return False

                # update the language model
                tm_entry_ok = True
                tok_words = transform_word(tok_array[prev_j])
                raw_word = transform_word(tok_array[prev_j])
                for k in range(prev_j + 1, j):
                    tok_words = tok_words + " " + transform_word(tok_array[k])
                    raw_word = raw_word + transform_word(tok_array[k])
                    if (is_categ(transform_word(tok_array[k - 1])) and
                            is_categ(transform_word(tok_array[k]))):
                        tm_entry_ok = False

                trans_raw_word_array.append(raw_word)

                # update variables
                i = i + 1
                prev_j = j

            self.train_word_array(trans_raw_word_array)

            # The sentence was successfully processed
            return True

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
        self.cursor.execute('select c from detokenize_ngram_counts where n=? limit 1', [word])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_all_counts(self):
        raise NotImplemented()

    def load_from_other_provider(self, provider):
        self.connection.execute('DROP TABLE IF EXISTS detokenize_ngram_counts')
        self.connection.execute('CREATE TABLE detokenize_ngram_counts (n text primary key not null, c int not null)')
        for key, value in provider.get_all_counts():
            self.cursor.execute('insert into detokenize_ngram_counts values (?, ?)', [' '.join(key), value])
        self.connection.commit()

