import os
import pandas as pd

from datasets import *


def get_number_of_features(df):
    datasets = df['dataset'].unique()
    n_features = []

    for i in range(len(datasets)):
        try:
            X, _ = globals()['load_' + datasets[i]]()
        except:
            X, _ = globals()['load_keel'](datasets[i])

        n_features.append(X.shape[1])

    return n_features


def load_df_and_datasets(k=None, b=None):
    frames = []

    for file_name in os.listdir('results'):
        frames.append(pd.read_csv(os.path.join('results', file_name), sep=',', skipinitialspace=True))

    df = pd.concat(frames)

    if k:
        df = df[df['k'] == k].reset_index()

    if b:
        df = df[(df['b'] == b) | (df['b'] == '-')].reset_index()

    datasets = df['dataset'].unique()
    features = get_number_of_features(df)

    return df, datasets[np.argsort(features)]


def print_datasets():
    _, datasets = load_df_and_datasets()

    print ('\\begin{table}\n'
           '\\caption{Datasets}\n'
           '\\centering\n'
           '\\setlength{\\tabcolsep}{2pt}\n'
           '\\begin{tabular}{cccc}\n'
           '\\hline\\noalign{\\smallskip}\n'
           '\\# & Name & Features & Samples \\\\\n'
           '\\noalign{\\smallskip}\n'
           '\\hline\n'
           '\\noalign{\\smallskip}')

    for i in range(len(datasets)):
        try:
            X, _ = globals()['load_' + datasets[i]]()
        except:
            X, _ = globals()['load_keel'](datasets[i])

        print '%d & %s & %d & %d \\\\' % (i + 1, datasets[i].replace('_', ' '), X.shape[1], X.shape[0])

    print ('\\hline\n'
           '\\end{tabular}\n'
           '\\end{table}\n')


def print_table(k, b=None):
    df, datasets = load_df_and_datasets(k, b)

    print ('\\begin{table}\n'
           '\\caption{k = ' + str(k) + '}\n'
           '\\centering\n'
           '\\setlength{\\tabcolsep}{2pt}\n'
           '\\resizebox{\\textwidth}{!}{\n'
           '\\begin{tabular}{cccccccccccccccccccccccccc}\n'
           '\\hline\\noalign{\\smallskip}\n'
           '\\# & RF & \\multicolumn{6}{l}{SVM} & \\multicolumn{6}{l}{kNN} & \\multicolumn{6}{l}{CART} & \\multicolumn{6}{l}{NaiveBayes} \\\\\n'
           '\\cmidrule(r){3-8} \\cmidrule(r){9-14} \\cmidrule(r){15-20} \\cmidrule(r){21-26}\n'
           ' & & & \\multicolumn{5}{l}{DS} & & \\multicolumn{5}{l}{DS} & & \\multicolumn{5}{l}{DS} & & \\multicolumn{5}{l}{DS} \\\\\n'
           '\\cmidrule(r){4-8} \\cmidrule(r){10-14} \\cmidrule(r){16-20} \\cmidrule(r){22-26}'
           ' & & RS & 0 & 0.25 & 0.5 & 0.75 & 1 & RS & 0 & 0.25 & 0.5 & 0.75 & 1 & RS & 0 & 0.25 & 0.5 & 0.75 & 1 & RS & 0 & 0.25 & 0.5 & 0.75 & 1 \\\\\n'
           '\\noalign{\\smallskip}\n'
           '\\hline\n'
           '\\noalign{\\smallskip}')

    for i in range(len(datasets)):
        row = '%d & ' % (i + 1)
        row += '%.2f & ' % (df[(df['dataset'] == datasets[i]) & (df['classifier'] == 'RandomForest')]['accuracy'])

        for classifier in ['SVM', 'kNN', 'DecisionTree', 'NaiveBayes']:
            rs_score = df[(df['dataset'] == datasets[i]) &
                          (df['classifier'] == classifier) &
                          (df['method'] == 'RandomSubspace')]['accuracy'].mean()

            row += '%.2f & ' % rs_score

            for alpha in ['0.0', '0.25', '0.5', '0.75', '1.0']:
                ds_score = df[(df['dataset'] == datasets[i]) &
                              (df['classifier'] == classifier) &
                              (df['method'] == 'DeterministicSubspace') &
                              (df['alpha'] == alpha)]['accuracy'].iloc[0]

                try:
                    ds_score = float(ds_score)
                    rs_score = float(rs_score)

                    if round(ds_score, 2) >= round(rs_score, 2):
                        row += '\cellcolor[gray]{0.8} '

                    row += '%.2f & ' % ds_score
                except ValueError:
                    row += '%s & ' % '-'

        print row[:-2] + ' \\\\'

    print ('\\hline\n'
           '\\end{tabular}\n'
           '}\n'
           '\\end{table}\n')


if __name__ == '__main__':
    print '\n\\subsection{Datasets}\n'

    print_datasets()

    for b in ['1', '2', '3']:
        print '\n\\subsection{Results, b = %s}\n' % b

        for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            print_table(k, b)
