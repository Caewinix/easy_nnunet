import sys
import re
import inspect
import argparse
from genericpy import generic


@generic
def run(*args, **kwargs):
    received = generic.receive()
    if isinstance(received, tuple):
        func, argv = received
        original_argv = sys.argv[1:]
        sys.argv[1:] = argv
    else:
        func = received
        original_argv = None
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    result = func(*args, **kwargs)
    if original_argv is not None:
        sys.argv[1:] = original_argv
    return result


@generic
def run_out(*args, **kwargs):
    return sys.exit(run[generic.receive()](*args, **kwargs))


class _TotallyMatchStr(str):
    def startswith(self, another):
        return self == another


class _TotallyMatchDict:
    def __init__(self, value):
        self._dict = value
    
    def __iter__(self):
        return (_TotallyMatchStr(option_string) for option_string in self._dict)
    
    def __getitem__(self, key):
        return self._dict[key]


class ArgumentParser(argparse.ArgumentParser):
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if name == '_option_string_actions' and inspect.currentframe().f_back.f_code.co_name == '_get_option_tuples':
            return _TotallyMatchDict(attr)
        return attr