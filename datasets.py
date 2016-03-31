import os
import urllib
import zipfile
import pandas as pd


def download(url, name=None, unpack=True):
    if not name:
        name = url.split('/')[-1]

    root_path = 'data'
    download_path = os.path.join(root_path, name)

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(download_path):
        urllib.urlretrieve(url, download_path)

    if unpack:
        with zipfile.ZipFile(download_path) as zip:
            zip.extractall(root_path)


def load_coil2000():
    download('http://sci2s.ugr.es/keel/dataset/data/classification/coil2000.zip')
    matrix = pd.read_csv('data/coil2000.dat', header=None, skiprows=90).as_matrix()
    X, y = matrix[:, :-1], matrix[:, -1]

    return X, y
