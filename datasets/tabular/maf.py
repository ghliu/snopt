import collections as co
import numpy as np
import os
import pathlib
import sktime.utils.load_data
import torch
import urllib.request
import tarfile

from .gas import GAS
from .miniboone import MINIBOONE


here = pathlib.Path(__file__).resolve().parent.parent.parent

def download():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'maf'
    loc = base_loc / 'maf.tar.gz'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)

    print('download from https://zenodo.org/record/1161203/files/data.tar.gz .....')
    urllib.request.urlretrieve('https://zenodo.org/record/1161203/files/data.tar.gz',
                               str(loc))

    def gas(tar):
        l = len("data/")
        for member in tar.getmembers():
            if member.path.startswith("data/gas"):
                member.path = member.path[l:]
                yield member

    def miniboone(tar):
        l = len("data/")
        for member in tar.getmembers():
            if member.path.startswith("data/miniboone"):
                member.path = member.path[l:]
                yield member

    with tarfile.open(loc, "r:gz") as tar:
        # tar.extractall(path=base_loc) # <---- TODO(Guan) use this if you wish to extract all datasets.
        tar.extractall(path=base_loc, members=gas(tar))
        tar.extractall(path=base_loc, members=miniboone(tar))

def get_data(dataset_name):

    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'maf'
    loc = base_loc / dataset_name

    if not os.path.exists(loc):
        download()

    return {
        'gas': GAS,
        'miniboone': MINIBOONE
    }.get(dataset_name)(loc)

