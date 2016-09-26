# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
from collections import defaultdict


def compute_deviations(differences_in_length):

    mean_deviation = sum(differences_in_length) / len(differences_in_length)
    std_deviation = 0
    for diff in differences_in_length:
        std_deviation += (diff - mean_deviation) * (diff - mean_deviation)
    std_deviation = math.sqrt(std_deviation / len(differences_in_length))

    return mean_deviation, std_deviation


def compute_mean_stddev_per_length(pairs_list):
    # Compute means
    mean_per_length = defaultdict(lambda: 0.0)
    samples_per_length = defaultdict(lambda: 0.0)
    for slen, dlen, diff in pairs_list:
        samples_per_length[slen] += 1.0
        mean_per_length[slen] += diff

    for k in samples_per_length:
        mean_per_length[k] = mean_per_length[k] / samples_per_length[k]

    # Compute standard deviations
    stddev_per_length = defaultdict(lambda: 0.0)
    for slen, dlen, diff in pairs_list:
        stddev_per_length[slen] += (diff - mean_per_length[slen]) * (diff - mean_per_length[slen])

    for k in samples_per_length.keys():
        stddev_per_length[k] = math.sqrt(stddev_per_length[k] / samples_per_length[k])

    # Return result
    return mean_per_length, stddev_per_length, samples_per_length
