import os
import sys
import pandas as pd


if len(sys.argv) > 1:
    root_path = sys.argv[1]
else:
    root_path = 'results'

if len(sys.argv) > 2:
    out_name = '%s.csv' % sys.argv[2]
else:
    out_name = 'results.csv'


frames = []

for file_name in os.listdir(root_path):
    frames.append(pd.read_csv(os.path.join(root_path, file_name), sep=',', skipinitialspace=True))

df = pd.concat(frames).reset_index(drop=True)

df.to_csv(out_name, index=False)
