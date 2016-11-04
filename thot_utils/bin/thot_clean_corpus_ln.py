# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""
from __future__ import division

import argparse
import io
import sys

from thot_utils.libs.math_functions import compute_deviations
from thot_utils.libs.math_functions import compute_mean_stddev_per_length
from thot_utils.libs.utils import split_string_to_words

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '-s',
    '--source-file',
    type=str,
    help='File with source text',
    required=True,
)
argparser.add_argument(
    '-t',
    '--target-file',
    type=str,
    help='File with target text',
    required=True,
)

argparser.add_argument(
    '-i',
    '--min-length',
    type=int,
    help='Minimum sentence length (1 by default)',
    default=1,
)

argparser.add_argument(
    '-a',
    '--max-length',
    type=int,
    help='Maximum sentence length (80 by default)',
    default=80,
)

argparser.add_argument(
    '-d',
    '--max-deviation',
    type=int,
    help='Maximum number of standard deviations allowed in the difference in length between the source and target '
         'sentences (4 by default)',
    default=80,
)

argparser.add_argument(
    '-m',
    '--min-samples',
    type=int,
    help='Minimum number of samples',
    default=10,
)


def main():
    cli_args = argparser.parse_args()

    source_fd = io.open(cli_args.source_file, 'r', encoding='utf-8')
    target_fd = io.open(cli_args.target_file, 'r', encoding='utf-8')

    # read parallel files line by line
    pairs_list = []
    for srcline, trgline in zip(source_fd, target_fd):
        src_word_len = len(split_string_to_words(srcline))
        trg_word_len = len(split_string_to_words(trgline))
        # Store sentence lengths
        pairs_list.append(
            (src_word_len, trg_word_len, src_word_len - trg_word_len)
        )

    # Compute statistics
    mean, stddev = compute_deviations(map(lambda x: x[2], pairs_list))
    mean_perl, stddev_perl, samples_perl = compute_mean_stddev_per_length(pairs_list)

    # Print line numbers
    for idx, slen, tlen, diff in enumerate(pairs_list, start=1):
        # Verify minimum and maximum length
        if (
            cli_args.min_length <= slen <= cli_args.max_length and
            cli_args.min_length <= tlen <= cli_args.max_length
        ):
            # Obtain difference in sentence length
            diff = slen - tlen
            # Obtain upper and lower limits for difference in sentence length
            if samples_perl[slen] >= cli_args.min_samples:
                uplim = mean_perl[slen] + cli_args.max_deviation * stddev_perl[slen]
                lolim = mean_perl[slen] - cli_args.max_deviation * stddev_perl[slen]
            else:
                uplim = mean + cli_args.max_deviation * stddev
                lolim = mean - cli_args.max_deviation * stddev

            # Verify difference in sentence length
            if uplim >= diff >= lolim:
                print(idx)
            else:
                print("lineno:", idx, ", slen:", slen, ", tlen:", tlen, file=sys.stderr)
        else:
            print("lineno:", idx, ", slen:", slen, ", tlen:", tlen, file=sys.stderr)


if __name__ == "__main__":
    main()
