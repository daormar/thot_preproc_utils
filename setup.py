import os
import re

import setuptools


def read_file(filename):
    """Open and a file, read it and return its contents."""
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path) as f:
        return f.read()


def get_version():
    """Extract and return version number from the packages '__init__.py'."""
    init_path = os.path.join('thot_utils', '__init__.py')
    content = read_file(init_path)
    match = re.search(r"__version__ = '([^']+)'", content, re.M)
    version = match.group(1)
    return version


metadata = dict(
    name='thot-utils',
    version=get_version(),
    description='Thot utils used in pre and post processing',
    install_requires=read_file('requirements.txt'),
    packages=setuptools.find_packages(
        include=['thot_utils', 'thot_utils.*'],
    ),
    include_package_data=True,
    zip_safe=True,
    package_data={
        '': ['requirements.txt'],
        'thot_utils.confs': ['*.ini'],
    },
    entry_points={
        'console_scripts': [
            'thot_categorize = thot_utils.bin.thot_categorize:main',
            'thot_decategorize = thot_utils.bin.thot_decategorize:main',
            'thot_clean_corpus_ln = thot_utils.bin.thot_clean_corpus_ln:main',
            'thot_lowercase = thot_utils.bin.thot_lowercase:main',
            'thot_recase = thot_utils.bin.thot_recase:main',
            'thot_recase_precalculate = thot_utils.bin.thot_recase_precalculate:main',
            'thot_tokenize = thot_utils.bin.thot_tokenize:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],
)


if __name__ == '__main__':
    setuptools.setup(**metadata)
