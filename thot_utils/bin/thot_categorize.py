# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""

import argparse
import codecs
import io
import sys

from thot_utils.libs.file_input import FileInput
from thot_utils.libs import thot_preproc

argparser = argparse.ArgumentParser(description=__doc__)
mutex_group = argparser.add_mutually_exclusive_group(required=True)
mutex_group.add_argument(
    '-f',
    '--file',
    type=str,
    help='File with text to be categorized (can be read from stdin)',
)

mutex_group.add_argument(
    '-s',
    '--stdin',
    action='store_true',
    help='Read model from standard input',
)


def main():
    cli_args = argparser.parse_args()
    if cli_args.stdin:
        fd = codecs.getreader('utf-8')(sys.stdin)
    else:
        fd = io.open(cli_args.file, 'r', encoding='utf-8')

    with FileInput(fd) as f:
        for line in f:
            categorized_line = thot_preproc.categorize(line)
            print categorized_line.encode('utf-8')
