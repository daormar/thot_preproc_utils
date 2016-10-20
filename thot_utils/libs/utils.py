# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re
from thot_utils.libs import config


split_regex = re.compile('[\u200b\s]+', flags=re.U)


def split_string_to_words(s):
    return [n for n in split_regex.split(s) if n]


def transform_word(word):
    if word.isdigit():
        if len(word) > 1:
            return config.number_str
        else:
            return config.digit_str
    elif is_number(word):
        return config.number_str
    elif is_alnum(word) and bool(config.digits.search(word)):
        return config.alfanum_str
    elif len(word) > 5:
        return config.common_word_str
    else:
        return word


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def is_alnum(s):
    res = config.alnum.match(s)
    if res is None:
        return False
    return True


def is_categ(word):
    if word in config.categ_set:
        return True
    else:
        return False
