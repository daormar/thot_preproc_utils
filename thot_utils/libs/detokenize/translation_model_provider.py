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


class TranslationModelFileProvider(object):
    def __init__(self, raw_fd, tokenized_fd):
        self.raw_fd = raw_fd
        self.tokenized_fd = tokenized_fd

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
                    self.increase_count(tok_words, raw_words)

                # update variables
                i = i + 1
                prev_j = j

            # The sentence was successfully processed
            return True

    def increase_count(self, src_words, trg_words):
        self.update_st_count(src_words, trg_words)
        self.update_s_count(src_words)

    def generate_sqlite(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

        self.connection.execute('DROP TABLE IF EXISTS detokenize_s_counts')
        self.connection.execute('CREATE TABLE detokenize_s_counts (t text primary key not null, c int not null)')

        self.connection.execute('DROP TABLE IF EXISTS detokenize_st_counts')
        self.connection.execute(
            'CREATE TABLE detokenize_st_counts (s text not null, t text not null, c int not null, PRIMARY KEY(s, t))'
        )

        # Read parallel files line by line
        for rline, tline in zip(self.raw_fd, self.tokenized_fd):
            raw_word_array = split_string_to_words(rline)
            tok_array = split_string_to_words(tline)
            # Process sentence
            retval = self.train_sent_tok(raw_word_array, tok_array)
            if not retval:
                print(
                    "Warning: something went wrong while training the translation model for sentence", file=sys.stderr)

        self.connection.commit()

    def update_s_count(self, key):
        self.cursor.execute('insert or ignore into detokenize_s_counts values (?, 0)', [key])
        self.cursor.execute('update detokenize_s_counts set c=c+1 where t=?', [key])

    def update_st_count(self, source, target):
        self.cursor.execute('insert or ignore into detokenize_st_counts values (?, ?, 0)', [source, target])
        self.cursor.execute('update detokenize_st_counts set c=c+1 where s=? and t=?', [source, target])


class TranslationModelDBProvider(object):
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
