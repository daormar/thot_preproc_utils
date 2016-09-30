# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""
import argparse
import codecs
import io
import sys

from thot_utils.libs import thot_preproc
from thot_utils.libs.file_input import FileInput

argparser = argparse.ArgumentParser(description=__doc__)
mutex_group = argparser.add_mutually_exclusive_group(required=True)
mutex_group.add_argument(
    '-f',
    '--file',
    type=str,
    help='File with text to be processed (can be read from stdin)',
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
            line = line.strip("\n")
            line = thot_preproc.lowercase(line)
            print line.encode("utf-8")


if __name__ == "__main__":
    main()
