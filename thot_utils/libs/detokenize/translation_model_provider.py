# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import abc
import itertools
import sqlite3
import sys
from collections import defaultdict

from thot_utils.libs.utils import is_categ
from thot_utils.libs.utils import split_string_to_words
from thot_utils.libs.utils import transform_word


class TranslationModelProviderInterface(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_targets(self, src_word):
        pass

    @abc.abstractmethod
    def get_target_count(self, src_words, trg_words):
        pass

    @abc.abstractmethod
    def get_source_count(self, src_words):
        pass

    @abc.abstractmethod
    def get_all_source_counts(self):
        pass

    @abc.abstractmethod
    def get_all_target_counts(self):
        pass


class TranslationModelFileProvider(TranslationModelProviderInterface):
    def __init__(self, raw_fd, tokenized_fd):
        self.raw_fd = raw_fd
        self.tokenized_fd = tokenized_fd
        self.st_counts = defaultdict(lambda: defaultdict(lambda: 0))
        self.s_counts = defaultdict(lambda: 0)

        self.run()

    def run(self):
        # Read parallel files line by line
        for rline, tline in zip(self.raw_fd, self.tokenized_fd):
            raw_word_array = split_string_to_words(rline)
            tok_array = split_string_to_words(tline)
            # Process sentence
            retval = self.train_sent_tok(raw_word_array, tok_array)
            if not retval:
                print(
                    "Warning: something went wrong while training the translation model for sentence", file=sys.stderr)

    def train_sent_tok(self, raw_word_array, tok_array):
        if (len(tok_array) > 0):
            # train translation model for sentence
            i = 0
            j = 0
            prev_j = 0
            error = False

            # Obtain transformed raw word array
            while i < len(raw_word_array):
                end = False
                str = ""

                # process current raw word
                while not end:
                    if raw_word_array[i] == str:
                        end = True
                    else:
                        if j >= len(tok_array):
                            error = True
                            end = True
                        else:
                            str = str + tok_array[j]
                            j = j + 1

                # Check that no errors were found while processing current raw word
                if error:
                    return False

                # update the translation model
                tm_entry_ok = True
                tok_words = transform_word(tok_array[prev_j])
                raw_word = transform_word(tok_array[prev_j])
                for k in range(prev_j + 1, j):
                    tok_words = tok_words + " " + transform_word(tok_array[k])
                    raw_word = raw_word + transform_word(tok_array[k])
                    if (is_categ(transform_word(tok_array[k - 1])) and
                            is_categ(transform_word(tok_array[k]))):
                        tm_entry_ok = False

                raw_words = raw_word

                if tm_entry_ok:
                    self.increase_count(tok_words, raw_words, 1)

                # update variables
                i = i + 1
                prev_j = j

            # The sentence was successfully processed
            return True

    def increase_count(self, src_words, trg_words, c):
        self.st_counts[src_words][trg_words] += c
        self.s_counts[src_words] += + c

    def get_targets(self, src_word):
        return self.st_counts[src_word].keys()

    def get_target_count(self, src_words, trg_words):
        return self.st_counts[src_words][trg_words]

    def get_source_count(self, src_words):
        return self.s_counts[src_words]

    def get_all_source_counts(self):
        for source, count in self.s_counts.items():
            yield source, count

    def get_all_target_counts(self):
        for source, targets_counts in self.st_counts.items():
            for target, count in targets_counts.items():
                yield source, target, count


class TranslationModelDBProvider(TranslationModelProviderInterface):
    def __init__(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

    def get_targets(self, src_word):
        self.cursor.execute('select t from detokenize_st_counts where s=?', [src_word])
        return [t for t, in self.cursor.fetchall()]

    def get_target_count(self, src_words, trg_words):
        self.cursor.execute('select c from detokenize_st_counts where s=? and t=? limit 1', [src_words, trg_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_source_count(self, src_words):
        self.cursor.execute('select c from detokenize_s_counts where t=? limit 1', [src_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_all_source_counts(self):
        raise NotImplemented()

    def get_all_target_counts(self):
        raise NotImplemented()

    def load_from_other_provider(self, provider):
        self.connection.execute('DROP TABLE IF EXISTS detokenize_s_counts')
        self.connection.execute('CREATE TABLE detokenize_s_counts (t text primary key not null, c int not null)')
        for key, value in provider.get_all_source_counts():
            self.cursor.execute('insert into detokenize_s_counts values (?, ?)', [key, value])
        self.connection.commit()

        self.connection.execute('DROP TABLE IF EXISTS detokenize_st_counts')
        self.connection.execute(
            'CREATE TABLE detokenize_st_counts (s text not null, t text not null, c int not null, PRIMARY KEY(s, t))')
        for source, target, count in provider.get_all_target_counts():
            self.cursor.execute('insert into detokenize_st_counts values (?, ?, ?)', [source, target, count])
        self.connection.commit()
