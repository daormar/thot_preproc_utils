# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""
import argparse
import io
import itertools

from thot_utils.libs import thot_preproc

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
    '--hypothesis-file',
    type=int,
    help='Minimum sentence length (1 by default)',
    default=1,
)


def main():
    cli_args = argparser.parse_args()

    sfile = io.open(cli_args.source_file, 'r', encoding='utf-8')
    tfile = io.open(cli_args.target_file, 'r', encoding='utf-8')
    ifile = io.open(cli_args.hypothesis_file, 'r', encoding='utf-8')

    for sline, tline, iline in zip(sfile, tfile, ifile):
        # Read source, target and hypothesis information
        sline = sline.strip('\n')
        tline = tline.strip('\n')
        iline = iline.strip('\n')

        decategorized_line = thot_preproc.decategorize(sline, tline, iline)
        print(decategorized_line.encode('utf-8'))


if __name__ == '__main__':
    main()

import nltk
nltk.word_tokenize('aa aa')
