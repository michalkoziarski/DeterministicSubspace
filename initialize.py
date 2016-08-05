import os
import database as db

from datasets import safe_load, get_names


RESULTS_DIR = 'results'
DATA_DIR = 'data'

print 'Creating results and data directories...'

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print 'Loading datasets...'

for dataset in get_names():
    print 'Loading %s...' % dataset

    try:
        safe_load(dataset)
    except:
        print '[Warning] Could not load %s dataset.' % dataset

print 'Creating database...'

db.create()
