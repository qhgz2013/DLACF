import os
from typing import List

from cornac.data import Reader
from cornac.utils import cache
from cornac.utils.download import get_cache_path


def _get_cache_dir():
    cache_dir = get_cache_path('ciaodvd')[0]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def load_feedback(min_user_freq, reader: Reader = None, fmt="UIR") -> List:
    if fmt == 'UIRT':
        fpath = cache(url='https://guoguibing.github.io/librec/datasets/CiaoDVD.zip',
                      unzip=True, cache_dir=_get_cache_dir(), relative_path='ratings2.txt')
        reader = Reader(min_user_freq=min_user_freq) if reader is None else reader
        return reader.read(fpath, sep=',', fmt=fmt)
    else:
        fpath = cache(url='https://guoguibing.github.io/librec/datasets/CiaoDVD.zip',
                      unzip=True, cache_dir=_get_cache_dir(), relative_path='review-ratings.txt')
        reader = Reader(min_user_freq=min_user_freq) if reader is None else reader
        return reader.read(fpath, sep=',')

def load_trust(reader: Reader = None) -> List:
    """Load the user-user trust information (undirected network)

    Parameters
    ----------
    reader: `obj:cornac.data.Reader`, default: None
        Reader object used to read the data.

    Returns
    -------
    data: array-like
        Data in the form of a list of tuples (user, user, 1).
    """
    fpath = cache(url='https://guoguibing.github.io/librec/datasets/CiaoDVD.zip',
                  unzip=True, cache_dir=_get_cache_dir(), relative_path='trusts.txt')
    reader = Reader() if reader is None else reader
    return reader.read(fpath, sep=',')
