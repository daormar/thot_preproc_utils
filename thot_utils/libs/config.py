# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import re

n = 2
lm_interp_prob = 0.5
common_word_str = "<common_word>"
number_str = "<number>"
digit_str = "<digit>"
alfanum_str = "<alfanum>"
unk_word_str = "<unk>"
eos_str = "<eos>"
bos_str = "<bos>"
categ_set = frozenset([common_word_str, number_str, digit_str, alfanum_str])
digits = re.compile('\d')
alnum = re.compile('[a-zA-Z0-9]+')
a_par = 7
maxniters = 100000
tm_smooth_prob = 0.000001

# xml annotation variables
grp_ann = "phr_pair_annot"
src_ann = "src_segm"
trg_ann = "trg_segm"
dic_patt = u"(<%s>)[ ]*(<%s>)(.+?)(<\/%s>)[ ]*(<%s>)(.+?)(<\/%s>)[ ]*(<\/%s>)" % (grp_ann,
                                                                                  src_ann, src_ann,
                                                                                  trg_ann, trg_ann,
                                                                                  grp_ann)
len_ann = "length_limit"
len_patt = u"(<%s>)[ ]*(\d+)[ ]*(</%s>)" % (len_ann, len_ann)

_annotation = re.compile(dic_patt + "|" + len_patt)
