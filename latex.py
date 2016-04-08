import os
import pandas as pd


def print_table(k, b=None, omega=None):
    frames = []

    for file_name in os.listdir('results'):
        frames.append(pd.read_csv(os.path.join('results', file_name), sep=',', skipinitialspace=True))

    df = pd.concat(frames)
    df = df[df['k'] == k].reset_index()

    print ('\\begin{table}\n'
           '\\caption{}\n'
           '\\centering\n'
           '\\setlength{\\tabcolsep}{6pt}\n'
           '\\begin{tabular}{c c c c c c c c c c c c c c c c c c c c c c c c c c}\n'
           '\\hline\\noalign{\\smallskip}\n'
           'Dataset & RandomForest  & \\multicolumn{6}{l}{SVM} & \\multicolumn{6}{l}{kNN} & \\multicolumn{6}{l}{CART} & \\multicolumn{6}{l}{NaiveBayes} \\\\\n'
           '\\noalign{\\smallskip}\n'
           '\\hline\n'
           '\\noalign{\\smallskip}')

    for dataset in sorted(df['dataset'].unique()):
        pass

    print ('\\hline\n'
           '\\end{tabular}\n'
           '\\end{table}\n')


print_table(5)
