# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sqlite3
import sys
from collections import Counter

from thot_utils.libs.utils import is_categ
from thot_utils.libs.utils import split_string_to_words
from thot_utils.libs.utils import transform_word


class TranslationModelFileProvider(object):
    def __init__(self, raw_fd, tokenized_fd):
        self.raw_fd = raw_fd
        self.tokenized_fd = tokenized_fd

    def train_sent_tok(self, counter_s_counts, counter_st_counts, raw_word_array, tok_array):
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
                    return

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
                    counter_s_counts[tok_words] += 1
                    counter_st_counts[tok_words, raw_words] += 1


                # update variables
                i = i + 1
                prev_j = j

    def increase_count(self, src_words, trg_words):
        self.update_st_count(src_words, trg_words)
        self.update_s_count(src_words)

    def generate_sqlite(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

        self.connection.execute('PRAGMA synchronous=OFF')
        self.connection.execute('PRAGMA cache_size=-2000000')

        self.connection.execute('DROP TABLE IF EXISTS detokenize_s_counts')
        self.connection.execute('CREATE TABLE detokenize_s_counts (t TEXT PRIMARY KEY NOT NULL, c INT NOT NULL)')

        self.connection.execute('DROP TABLE IF EXISTS detokenize_st_counts')
        self.connection.execute(
            'CREATE TABLE detokenize_st_counts (s TEXT NOT NULL, t TEXT NOT NULL, c INT NOT NULL, PRIMARY KEY(s, t))'
        )

        # Read parallel files line by line
        counter_s_counts = Counter()
        counter_st_counts = Counter()
        for idx, (rline, tline) in enumerate(zip(self.raw_fd, self.tokenized_fd)):
            raw_word_array = split_string_to_words(rline)
            tok_array = split_string_to_words(tline)
            # Process sentence
            self.train_sent_tok(counter_s_counts, counter_st_counts, raw_word_array, tok_array)

            if idx % 100000 == 0:
                print(idx)
                self.update_s_count(counter_s_counts)
                self.update_st_count(counter_st_counts)
                counter_s_counts = Counter()
                counter_st_counts = Counter()

        if counter_s_counts:
            self.update_s_count(counter_s_counts)

        if counter_st_counts:
            self.update_st_count(counter_st_counts)

        self.connection.commit()

    def update_s_count(self, counter):
        items = counter.items()
        keys = [(k[0],) for k in items]
        self.cursor.executemany('INSERT OR IGNORE INTO detokenize_s_counts VALUES (?, 0)', keys)
        self.cursor.executemany('UPDATE detokenize_s_counts SET c=c+?2 WHERE t=?1', items)

    def update_st_count(self, counter):
        items = counter.items()
        items = [(s, t, c) for (s, t), c in items]
        self.cursor.executemany('INSERT OR IGNORE INTO detokenize_st_counts VALUES (?, ?, 0)', counter.keys())
        self.cursor.executemany('UPDATE detokenize_st_counts SET c=c+?3 WHERE s=?1 AND t=?2', items)


class TranslationModelDBProvider(object):
    def __init__(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

    def get_targets(self, src_word):
        self.cursor.execute('SELECT t FROM detokenize_st_counts WHERE s=?', [src_word])
        return [t for t, in self.cursor.fetchall()]

    def get_target_count(self, src_words, trg_words):
        self.cursor.execute('SELECT c FROM detokenize_st_counts WHERE s=? AND t=? LIMIT 1', [src_words, trg_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_source_count(self, src_words):
        self.cursor.execute('SELECT c FROM detokenize_s_counts WHERE t=? LIMIT 1', [src_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0
