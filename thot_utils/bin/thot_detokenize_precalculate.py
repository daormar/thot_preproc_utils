# Author: Daniel Ortiz Mart\'inez

import argparse
import io

from thot_utils.libs.detokenize.language_model_file_provider import LanguageModelDBProvider, LanguageModelFileProvider
from thot_utils.libs.detokenize.translation_model_provider import TranslationModelDBProvider
from thot_utils.libs.detokenize.translation_model_provider import TranslationModelFileProvider

argparser = argparse.ArgumentParser(description=__doc__)

argparser.add_argument(
    '-r',
    '--raw',
    type=str,
    help='File with raw text in the language of interest.',
    required=True,
)

argparser.add_argument(
    '-t',
    '--tokenized',
    type=str,
    help='File with tokenized text in the language of interest.',
    required=True,
)


##################################################
def main():
    cli_args = argparser.parse_args()

    raw_fd = io.open(cli_args.raw, 'r', encoding='utf-8')
    tokenized_fd = io.open(cli_args.tokenized, 'r', encoding='utf-8')
    translation_model_provider = TranslationModelFileProvider(
        raw_fd=raw_fd, tokenized_fd=tokenized_fd
    )
    db_translation_model_provider = TranslationModelDBProvider('%s.sqlite' % cli_args.raw)
    db_translation_model_provider.load_from_other_provider(translation_model_provider)

    raw_fd = io.open(cli_args.raw, 'r', encoding='utf-8')
    tokenized_fd = io.open(cli_args.tokenized, 'r', encoding='utf-8')
    language_model_provider = LanguageModelFileProvider(raw_fd=raw_fd, tokenized_fd=tokenized_fd, ngrams_length=2)
    db_language_model_provider = LanguageModelDBProvider('%s.sqlite' % cli_args.raw)
    db_language_model_provider.load_from_other_provider(language_model_provider)


if __name__ == "__main__":
    main()
