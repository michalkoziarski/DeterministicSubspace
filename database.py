import os
import datetime
import sqlite3
import pandas as pd


RESULTS_DIR = 'results'
DB_PATH = 'results.db'


def execute(command, fetch=False):
    conn = sqlite3.connect(os.path.join(RESULTS_DIR, DB_PATH))

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
        execute('''CREATE TABLE trials (timestamp text, dataset text, fold text,
                classifier text, method text, measure text, k text, alpha text)''')


def insert(dataset, fold, classifier, method, measure, k, alpha):
    timestamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

    execute('INSERT INTO trials VALUES ("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s")' %
            (timestamp, dataset, fold, classifier, method, measure, k, alpha))


def export(path='results.csv'):
    rows = execute('SELECT * FROM trials', fetch=True)

    df = pd.DataFrame(rows, columns=['timestamp', 'dataset', 'fold', 'classifier', 'method', 'measure', 'k', 'alpha'])
    df.to_csv(os.path.join(RESULTS_DIR, path), index=False)
