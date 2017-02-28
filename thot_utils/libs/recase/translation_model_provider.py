# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sqlite3
from collections import Counter

from thot_utils.libs.thot_preproc import lowercase
from thot_utils.libs.utils import split_string_to_words


class TranslationModelFileProvider(object):
    def __init__(self, raw_fd):
        self.fd = raw_fd

    def train_sent_rec(self, raw_word_array, lc_word_array):
        for raw_word, lc_word in zip(raw_word_array, lc_word_array):
            self.increase_count(lc_word, raw_word)

    def increase_count(self, src_words, trg_words):
        self.update_st_count(src_words, trg_words)
        self.update_s_count(src_words)

    def generate_sqlite(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

        self.connection.execute('DROP TABLE IF EXISTS s_counts')
        self.connection.execute('DROP TABLE IF EXISTS st_counts')
        self.connection.execute('CREATE TABLE s_counts (t TEXT PRIMARY KEY NOT NULL, c INT NOT NULL)')
        self.connection.execute(
            'CREATE TABLE st_counts (s TEXT NOT NULL, t TEXT NOT NULL, c INT NOT NULL, PRIMARY KEY(s, t))')
        self.connection.execute('PRAGMA synchronous=OFF')
        self.connection.execute('PRAGMA count_changes=OFF')
        counter_s_count = Counter()
        counter_st_count = Counter()
        for idx, line in enumerate(self.fd):
            raw_word_array = split_string_to_words(line)
            lc_word_array = split_string_to_words(lowercase(line))
            for s, t in zip(raw_word_array, lc_word_array):
                counter_s_count[s] += 1
                counter_st_count[s, t] += 1
            if idx == 100000:
                self.update_s_count(counter_s_count)
                self.update_st_count(counter_st_count)
                counter_s_count = Counter()
                counter_st_count = Counter()

        if counter_s_count:
            self.update_s_count(counter_s_count)
        if counter_st_count:
            self.update_st_count(counter_st_count)

        self.connection.commit()

    def update_s_count(self, counter):
        items = counter.items()
        keys = [(k[0],) for k in items]
        self.cursor.executemany('INSERT OR IGNORE INTO s_counts VALUES (?, 0)', keys)
        self.cursor.executemany('UPDATE s_counts SET c=c+?2 WHERE t=?1', items)

    def update_st_count(self, counter):
        items = counter.items()
        items = [(s, t, c) for (s, t), c in items]
        self.cursor.executemany('INSERT OR IGNORE INTO st_counts VALUES (?, ?, 0)', counter.keys())
        self.cursor.executemany('UPDATE st_counts SET c=c+?3 WHERE s=?1 AND t=?2', items)


class TranslationModelDBProvider(object):
    def __init__(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

    def get_targets(self, src_word):
        self.cursor.execute('SELECT t FROM st_counts WHERE s=?', [src_word])
        return [t for t, in self.cursor.fetchall()]

    def get_target_count(self, src_words, trg_words):
        self.cursor.execute('SELECT c FROM st_counts WHERE s=? AND t=? LIMIT 1', [src_words, trg_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_source_count(self, src_words):
        self.cursor.execute('SELECT c FROM s_counts WHERE t=? LIMIT 1', [src_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0
