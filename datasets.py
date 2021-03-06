import os
import urllib
import zipfile
import numpy as np


def download_dataset(url, unpack=True):
    name = url.split('/')[-1]
    root_path = 'data'
    download_path = os.path.join(root_path, name)

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(download_path):
        urllib.urlretrieve(url, download_path)

    if unpack:
        if name.endswith('.zip'):
            with zipfile.ZipFile(download_path) as zip:
                zip.extractall(root_path)
        else:
            raise Exception('Unrecognized file type.')


def apply_encoding(X, y, encode_features=True):
    from sklearn import preprocessing

    y = preprocessing.LabelEncoder().fit(y).transform(y)

    if encode_features:
        encoded = []

        for i in range(X.shape[1]):
            try:
                float(X[0, i])
                encoded.append(X[:, i])
            except:
                encoded.append(preprocessing.LabelEncoder().fit_transform(X[:, i]))
        X = np.transpose(encoded)

    return X.astype(np.float32), y.astype(np.float32)


def load(file_name, url=None, download=True, skiprows=0, unpack=True, encode=True, separator=',', start=0,
         skipcols=None, nrows=None):
    from pandas import read_csv

    if download and url is not None:
        download_dataset(url, unpack=unpack)

    df = read_csv(os.path.join('data', file_name), header=None, skiprows=skiprows, skipinitialspace=True,
                  error_bad_lines=False, sep=separator, na_values='?', nrows=nrows)

    if skipcols:
        df = df.drop(df.columns[skipcols], axis=1)

    matrix = df.dropna().as_matrix()
    X, y = matrix[:, start:-1], matrix[:, -1]

    return apply_encoding(X, y, encode)


def load_keel(name, encode=True, download=True):
    base_url = 'http://sci2s.ugr.es/keel/dataset/data/classification/'
    url = '%s%s.zip' % (base_url, name)
    file_name = '%s.dat' % name

    if download:
        download_dataset(url)

    metadata_lines = 0

    with open(os.path.join('data', file_name)) as f:
        for line in f:
            if line.startswith('@'):
                metadata_lines += 1
            else:
                break

    return load(file_name, url, download=download, skiprows=metadata_lines, encode=encode)


def load_winequality(download=True):
    X_red, y_red = load_keel('winequality-red', download=download)
    X_white, y_white = load_keel('winequality-white')

    return np.concatenate((X_red, X_white)), np.concatenate((y_red, y_white))


def load_chronic_kidney_disease(download=True):
    url = 'https://github.com/michalkoziarski/datasets/raw/master/chronic_kidney_disease_full.arff'
    file_name = 'chronic_kidney_disease_full.arff'

    return load(file_name, url, unpack=False, skiprows=145, download=download)


def load_biodegradation(download=True):
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
    file_name = 'biodeg.csv'

    return load(file_name, url, unpack=False, separator=';', download=download)


def load_mice_protein_expression(download=True):
    from pandas import read_excel

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
    file_name = 'Data_Cortex_Nuclear.xls'

    if download:
        download_dataset(url, unpack=False)

    matrix = read_excel(os.path.join('data', file_name)).dropna().as_matrix()
    X, y = matrix[:, 1:-1], matrix[:, -1]

    return apply_encoding(X, y)


def load_musk(download=True):
    url = 'https://github.com/michalkoziarski/datasets/raw/master/musk.data.zip'
    file_name = 'musk.data'

    return load(file_name, url, start=2, download=download)


def load_isolet(download=True):
    url = 'https://github.com/michalkoziarski/datasets/raw/master/isolet.data.zip'
    file_name = 'isolet.data'

    return load(file_name, url, download=download)


def load_internet_ads(download=True):
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data'
    file_name = 'ad.data'

    return load(file_name, url, unpack=False, download=download)


def load_mutants(download=True):
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/p53/p53_new_2012.zip'
    file_name = os.path.join('Data Sets', 'K9.data')

    return load(file_name, url, skipcols=[-1], download=download)


def safe_load(dataset, download=True):
    if 'load_%s' % dataset in globals():
        return globals()['load_%s' % dataset](download=download)
    else:
        try:
            return load_keel(dataset, download=download)
        except:
            raise AttributeError('Dataset with specified name (%s) could not be loaded.' % dataset)


def quick_load(dataset):
    return safe_load(dataset, download=False)


def get_names():
    names = []

    with open('datasets.txt', 'r') as f:
        for dataset in f:
            names.append(dataset.rstrip())

    return names


def load_all():
    datasets = {}

    for name in get_names():
        datasets[name] = safe_load(name)

    return datasets
