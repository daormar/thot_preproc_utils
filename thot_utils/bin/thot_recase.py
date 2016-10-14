# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""
import argparse
import codecs
import sys

import io
from thot_utils.libs import thot_preproc
from thot_utils.libs.file_input import FileInput
from thot_utils.libs.language_model_file_provider import LanguageModelDBProvider
from thot_utils.libs.translation_model_file_provider import TranslationModelDBPrivider

argparser = argparse.ArgumentParser(description=__doc__)

argparser.add_argument(
    '-r',
    '--raw',
    type=str,
    help='File with raw text in the language of interest.',
    required=True,
)

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

    db_translation_model_provider = TranslationModelDBPrivider('%s.sqlite' % cli_args.raw)

    tmodel = thot_preproc.TransModel(
        model_provider=db_translation_model_provider
    )
    db_language_model_provider = LanguageModelDBProvider('%s.sqlite' % cli_args.raw)
    lmodel = thot_preproc.LangModel(db_language_model_provider, ngrams_length=2)

    weights = [0, 0, 0, 1]
    decoder = thot_preproc.Decoder(tmodel, lmodel, weights)

    print >> sys.stderr, "Recasing..."
    if cli_args.stdin:
        fd = codecs.getreader('utf-8')(sys.stdin)
    else:
        fd = io.open(cli_args.file, 'r', encoding='utf-8')

    with FileInput(fd) as f:
        for line in f:
            decoder.recase([line], False)


if __name__ == "__main__":
    main()
