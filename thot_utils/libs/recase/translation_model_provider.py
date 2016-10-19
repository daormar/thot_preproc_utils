# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import abc
from collections import defaultdict

import sqlite3
from thot_utils.libs.thot_preproc import lowercase


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
    def __init__(self, fd):
        self.fd = fd
        self.st_counts = defaultdict(lambda: defaultdict(lambda: 0))
        self.s_counts = defaultdict(lambda: 0)

        self.run()

    def run(self):
        for line in self.fd:
            line = line.strip("\n")
            raw_word_array = line.split()
            lc_word_array = lowercase(line).split()
            self.train_sent_rec(raw_word_array, lc_word_array)

    def train_sent_rec(self, raw_word_array, lc_word_array):
        for raw_word, lc_word in zip(raw_word_array, lc_word_array):
            self.increase_count(lc_word, raw_word, 1)

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
        for source, count in self.s_counts.iteritems():
            yield source, count

    def get_all_target_counts(self):
        for source, targets_counts in self.st_counts.iteritems():
            for target, count in targets_counts.iteritems():
                yield source, target, count


class TranslationModelDBProvider(TranslationModelProviderInterface):
    def __init__(self, filename):
        self.connection = sqlite3.connect(filename)
        self.cursor = self.connection.cursor()

    def get_targets(self, src_word):
        self.cursor.execute('select t from st_counts where s=?', [src_word])
        return [t for t, in self.cursor.fetchall()]

    def get_target_count(self, src_words, trg_words):
        self.cursor.execute('select c from st_counts where s=? and t=? limit 1', [src_words, trg_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_source_count(self, src_words):
        self.cursor.execute('select c from s_counts where t=? limit 1', [src_words])
        rows = self.cursor.fetchall()
        if rows:
            return rows[0][0]
        return 0

    def get_all_source_counts(self):
        raise NotImplemented()

    def get_all_target_counts(self):
        raise NotImplemented()

    def load_from_other_provider(self, provider):
        self.connection.execute('CREATE TABLE s_counts (t text primary key not null, c int not null)')
        for key, value in provider.get_all_source_counts():
            self.cursor.execute('insert into s_counts values (?, ?)', [key, value])
        self.connection.commit()

        self.connection.execute(
            'CREATE TABLE st_counts (s text not null, t text not null, c int not null, PRIMARY KEY(s, t))')
        for source, target, count in provider.get_all_target_counts():
            self.cursor.execute('insert into st_counts values (?, ?, ?)', [source, target, count])
        self.connection.commit()

