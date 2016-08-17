import os
import datetime
import sqlite3


DATABASE_DIR = 'results'
RESULTS_PATH = 'results.db'
PENDING_PATH = 'pending.db'
ACTIVE_PATH = 'active.db'


def add_to_pending(trial, check_presence=True):
    if check_presence and select(trial, database=RESULTS_PATH) is not None:
        return False

    insert(trial, database=PENDING_PATH)

    return True


def pull_pending():
    connection = connect(database=PENDING_PATH, exclusive=True)
    trial = select(connection=connection, fetch='one')

    if trial is not None:
        insert(trial, database=ACTIVE_PATH)
        delete(trial, connection=connection)

    connection.commit()
    connection.close()

    return trial


def submit_result(trial, score):
    delete(trial, database=ACTIVE_PATH)

    trial.update({'score': round(float(score), 4)})

    insert(trial, database=RESULTS_PATH)


def clear_active():
    trials = select(database=ACTIVE_PATH, fetch='all')

    for trial in trials:
        delete(trial, ACTIVE_PATH)
        insert(trial, PENDING_PATH)


def export(database=RESULTS_PATH, path='results.csv'):
    import pandas as pd

    trials = select(database=database, fetch='all')
    df = pd.DataFrame(trials, columns=_columns(score=(database == RESULTS_PATH)))
    df.to_csv(os.path.join(DATABASE_DIR, path), index=False)


def select(trial=None, database=None, connection=None, fetch='one'):
    assert fetch in ['one', 'all']

    command = 'SELECT * FROM trials'

    if trial is not None:
        command += ' %s' % _where(trial)

    return execute(command, database=database, connection=connection, fetch=fetch)


def insert(trial, database=None, connection=None):
    trial['timestamp'] = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())

    if database == RESULTS_PATH:
        trial['score'] = round(float(trial['score']), 4)

    execute('INSERT INTO trials (%s) VALUES (%s)' %
            (', '.join(trial.keys()), ', '.join(['"%s"' % value for value in trial.values()])),
            database=database, connection=connection)


def delete(trial, database=None, connection=None):
    return execute('DELETE FROM trials %s' % _where(trial), database=database, connection=connection)


def execute(command, connection=None, database=None, fetch='none'):
    assert connection is not None or database is not None
    assert fetch in ['none', 'one', 'all']

    if connection is None:
        conn = connect(database)
    else:
        conn = connection

    cursor = conn.cursor()
    cursor.execute(command)

    if fetch == 'one':
        result = cursor.fetchone()
    elif fetch == 'all':
        result = cursor.fetchall()
    else:
        result = None

    if connection is None:
        conn.commit()
        conn.close()

    return result


def connect(database, exclusive=False, timeout=600.0):
    connection = sqlite3.connect(os.path.join(DATABASE_DIR, database), timeout=timeout)
    connection.row_factory = _dict_factory

    if exclusive:
        connection.isolation_level = 'EXCLUSIVE'
        connection.execute('BEGIN EXCLUSIVE')

    return connection


def create(database):
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    if not os.path.exists(os.path.join(DATABASE_DIR, database)):
        if database == RESULTS_PATH:
            columns = _columns(score=True)
        else:
            columns = _columns(score=False)

        columns = ['%s text' % column for column in columns]
        columns = ', '.join(columns)

        execute('CREATE TABLE trials (%s)' % columns, database=database)


def _columns(score=True, timestamp=True):
    columns = ['timestamp', 'dataset', 'fold', 'classifier', 'method', 'measure', 'k', 'n', 'alpha', 'score']

    if not score:
        columns = columns[:-1]

    if not timestamp:
        columns = columns[1:]

    return columns


def _where(trial):
    selector = 'WHERE '

    for k, v in trial.iteritems():
        if k == 'timestamp' or k == 'score':
            continue

        selector += '%s="%s" AND ' % (k, v)

    selector = selector[:-5]

    return selector


def _dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d
