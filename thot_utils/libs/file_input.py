# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


class FileInput(object):
    def __init__(self, fd):
        self.fd = fd

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.fd.close()

    def __iter__(self):
        return self

    def next(self):
        try:
            line = self.fd.readline()
        except KeyboardInterrupt:
            line = None
        if line is None or line == "":
            raise StopIteration
        return line
