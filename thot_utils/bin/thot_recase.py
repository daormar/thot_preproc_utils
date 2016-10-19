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
from thot_utils.libs import recase

argparser = argparse.ArgumentParser(description=__doc__)

argparser.add_argument(
    '-r',
    '--sqlite',
    type=str,
    help='File with precomputed date from text in the language of interest.',
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

    db_translation_model_provider = recase.TranslationModelDBProvider(cli_args.sqlite)

    tmodel = thot_preproc.TransModel(
        model_provider=db_translation_model_provider
    )
    db_language_model_provider = recase.LanguageModelDBProvider(cli_args.sqlite)
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
            line = decoder.recase([line], False)
            print line.encode("utf-8")


if __name__ == "__main__":
    main()
