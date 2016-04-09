import os
import urllib
import zipfile
import numpy as np
import pandas as pd

from sklearn import preprocessing


def download(url, unpack=True):
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


def load(url, file_name, skiprows=0, unpack=True, encode=True, separator=',', start=0, skipcols=None, nrows=None):
    download(url, unpack=unpack)
    df = pd.read_csv(os.path.join('data', file_name), header=None, skiprows=skiprows, skipinitialspace=True,
                     error_bad_lines=False, sep=separator, na_values='?', nrows=nrows)

    if skipcols:
        df = df.drop(df.columns[skipcols], axis=1)

    matrix = df.dropna().as_matrix()
    X, y = matrix[:, start:-1], matrix[:, -1]

    return apply_encoding(X, y, encode)


def load_keel(name, encode=True):
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

    return load(url, file_name, skiprows=metadata_lines, encode=encode)


def load_winequality():
    X_red, y_red = load_keel('winequality-red')
    X_white, y_white = load_keel('winequality-white')

    return np.concatenate((X_red, X_white)), np.concatenate((y_red, y_white))


def load_chronic_kidney_disease():
    url = 'https://s3.amazonaws.com/michalkoziarski/chronic_kidney_disease_full.arff'
    file_name = 'chronic_kidney_disease_full.arff'

    return load(url, file_name, unpack=False, skiprows=145)


def load_biodegradation():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
    file_name = 'biodeg.csv'

    return load(url, file_name, unpack=False, separator=';')


def load_mice_protein_expression():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
    file_name = 'Data_Cortex_Nuclear.xls'

    download(url, unpack=False)

    matrix = pd.read_excel(os.path.join('data', 'Data_Cortex_Nuclear.xls')).dropna().as_matrix()
    X, y = matrix[:, 1:-1], matrix[:, -1]

    return apply_encoding(X, y)


def load_musk():
    url = 'https://s3.amazonaws.com/michalkoziarski/musk.data.zip'
    file_name = 'musk.data'

    return load(url, file_name, start=2)


def load_isolet():
    url = 'https://s3.amazonaws.com/michalkoziarski/isolet.data.zip'
    file_name = 'isolet.data'

    return load(url, file_name)


def load_internet_ads():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/internet_ads/ad.data'
    file_name = 'ad.data'

    return load(url, file_name, unpack=False)


def load_mutants():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/p53/p53_new_2012.zip'
    file_name = os.path.join('Data Sets', 'K9.data')

    return load(url, file_name, skipcols=[-1])


def load_all():
    return {
        'coil2000': load_keel('coil2000'),
        'movement_libras': load_keel('movement_libras'),
        'optdigits': load_keel('optdigits'),
        'segment': load_keel('segment'),
        'sonar': load_keel('sonar'),
        'spambase': load_keel('spambase'),
        'splice': load_keel('splice'),
        'texture': load_keel('texture'),
        'vowel': load_keel('vowel'),
        'vehicle': load_keel('vehicle'),
        'mushroom': load_keel('mushroom'),
        'automobile': load_keel('automobile'),
        'ionsphere': load_keel('ionsphere'),
        'thyroid': load_keel('thyroid'),
        'wdbc': load_keel('wdbc'),
        'spectfheart': load_keel('spectfheart'),
        'dermatology': load_keel('dermatology'),
        'ring': load_keel('ring'),
        'winequality': load_winequality(),
        'chronic_kidney_disease': load_chronic_kidney_disease(),
        'biodegradation': load_biodegradation(),
        'mice_protein_expression': load_mice_protein_expression(),
        'musk': load_musk(),
        'isolet': load_isolet(),
        'internet_ads': load_internet_ads(),
        'mutants': load_mutants()
    }
