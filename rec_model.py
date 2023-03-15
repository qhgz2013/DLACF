import numpy as _np
from distutils.version import LooseVersion as _LooseVersion

# Fix alias compatibility if numpy >= 1.24
if _np.__version__ >= _LooseVersion('1.24'):
    print('Numpy compatibility warning: type alias is removed since 1.24')
    for _alias in ['int', 'float', 'bool']:
        setattr(_np, _alias, getattr(_np, _alias + '_', None))


class RecModel:
    def init(self):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def train(self, max_itr: int, max_itr2: int, reset_param: bool = True):
        raise NotImplementedError
