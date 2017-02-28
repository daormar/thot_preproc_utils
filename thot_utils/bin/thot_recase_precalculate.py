# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""
import argparse
import io

from thot_utils.libs import recase

argparser = argparse.ArgumentParser(description=__doc__)

argparser.add_argument(
    '-r',
    '--raw',
    type=str,
    help='File with raw text in the language of interest.',
    required=True,
)

argparser.add_argument(
    '-o',
    '--output',
    type=str,
    help='Sqlite file where we will write output.',
    required=True,
)


def main():
    cli_args = argparser.parse_args()

    raw_fd = io.open(cli_args.raw, 'r', encoding='utf-8')
    translation_model_provider = recase.TranslationModelFileProvider(raw_fd=raw_fd)
    translation_model_provider.generate_sqlite(filename=cli_args.output)

    raw_fd = io.open(cli_args.raw, 'r', encoding='utf-8')
    language_model_provider = recase.LanguageModelFileProvider(raw_fd, ngrams_length=2)
    language_model_provider.generate_sqlite(filename=cli_args.output)

if __name__ == "__main__":
    main()
