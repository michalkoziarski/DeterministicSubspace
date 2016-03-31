import os
import urllib
import zipfile
import pandas as pd


def download(url, unpack=True):
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


def load(url, file_name, skiprows=0, unpack=True):
    download(url, unpack=unpack)
    matrix = pd.read_csv(os.path.join('data', file_name), header=None, skiprows=skiprows).as_matrix()
    X, y = matrix[:, :-1], matrix[:, -1]

    return X, y


def load_keel(name):
    base_url = 'http://sci2s.ugr.es/keel/dataset/data/classification/'
    url = '%s%s.zip' % (base_url, name)
    file_name = '%s.dat' % name
    download(url)
    metadata_lines = 0

    with open(os.path.join('data', file_name)) as f:
        for line in f:
            if line.startswith('@'):
                metadata_lines += 1
            else:
                break

    return load(url, file_name, skiprows=metadata_lines)


def load_all():
    return {
        'coil2000': load_keel('coil2000'),
        'movement_libras': load_keel('movement_libras'),
        'optdigits': load_keel('optdigits'),
        'segment': load_keel('segment'),
        'sonar': load_keel('sonar'),
        'spambase': load_keel('spambase'),
        'splice': load_keel('splice'),
        'texture': load_keel('texture')
    }
