import os
import datetime
import sqlite3


RESULTS_DIR = 'results'
DB_PATH = 'results.db'


def execute(command, fetch=False):
    conn = sqlite3.connect(os.path.join(RESULTS_DIR, DB_PATH), timeout=3600.)

    cursor = conn.cursor()
    cursor.execute(command)

    if fetch:
        rows = cursor.fetchall()
    else:
        rows = []

    conn.commit()
    conn.close()

    return rows


def create():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(os.path.join(RESULTS_DIR, DB_PATH)):
        execute('''CREATE TABLE trials (timestamp text, dataset text, fold text, classifier text,
                method text, measure text, k text, n text, alpha text, score text)''')


def insert(dataset, fold, classifier, method, measure, k, n, alpha, score):
    timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

    if not isinstance(score, basestring):
        score = round(score, 4)

    execute('INSERT INTO trials VALUES ("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' %
            (timestamp, dataset, fold, classifier, method, measure, k, n, alpha, score))


def reserve(dataset, fold, classifier, method, measure, k, n, alpha):
    conn = sqlite3.connect(os.path.join(RESULTS_DIR, DB_PATH), timeout=3600.)
    conn.isolation_level = 'EXCLUSIVE'
    conn.execute('BEGIN EXCLUSIVE')

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM trials WHERE dataset="%s" AND fold="%s" AND classifier="%s" AND method="%s" '
                   'AND measure="%s" AND k="%s" AND n="%s" AND alpha="%s"' %
                   (dataset, fold, classifier, method, measure, k, n, alpha))

    rows = cursor.fetchall()

    if len(rows) > 0:
        result = False
    else:
        result = True
        timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

        cursor.execute('INSERT INTO trials VALUES ("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "-")' %
                       (timestamp, dataset, fold, classifier, method, measure, k, n, alpha))

    conn.commit()
    conn.close()

    return result


def update(dataset, fold, classifier, method, measure, k, n, alpha, score):
    score = round(score, 4)

    execute('UPDATE trials SET score="%s" WHERE dataset="%s" AND fold="%s" AND classifier="%s" AND method="%s" '
            'AND measure="%s" AND k="%s" AND n="%s" AND alpha="%s" AND score="-"' %
            (score, dataset, fold, classifier, method, measure, k, n, alpha))


def export(path='results.csv'):
    import pandas as pd

    rows = execute('SELECT * FROM trials', fetch=True)

    df = pd.DataFrame(rows, columns=['timestamp', 'dataset', 'fold', 'classifier', 'method',
                                     'measure', 'k', 'n', 'alpha', 'score'])
    df.to_csv(os.path.join(RESULTS_DIR, path), index=False)
