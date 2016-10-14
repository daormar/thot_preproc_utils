# -*- coding:utf-8 -*-
# Author: Daniel Ortiz Mart'inez
"""
Some description
"""
import argparse
import io

from thot_utils.libs.language_model_file_provider import LanguageModelDBProvider, LanguageModelFileProvider

argparser = argparse.ArgumentParser(description=__doc__)

argparser.add_argument(
    '-r',
    '--raw',
    type=str,
    help='File with raw text in the language of interest.',
    required=True,
)


def main():
    cli_args = argparser.parse_args()

    fd = io.open(cli_args.raw, 'r', encoding='utf-8')
    translation_model_provider = TranslationModelFileProvider(fd)
    db_translation_model_provider = TranslationModelDBPrivider('%s.sqlite' % cli_args.raw)
    db_translation_model_provider.load_from_other_provider(translation_model_provider)

    fd = io.open(cli_args.raw, 'r', encoding='utf-8')
    language_model_provider = LanguageModelFileProvider(fd, ngrams_length=2)
    db_language_model_provider = LanguageModelDBProvider('%s.sqlite' % cli_args.raw)
    db_language_model_provider.load_from_other_provider(language_model_provider)


if __name__ == "__main__":
    main()
