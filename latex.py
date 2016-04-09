import os
import pandas as pd


def print_table(k, b=None, omega=None):
    frames = []

    for file_name in os.listdir('results'):
        frames.append(pd.read_csv(os.path.join('results', file_name), sep=',', skipinitialspace=True))

    df = pd.concat(frames)
    df = df[df['k'] == k].reset_index()

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

    datasets = sorted(df['dataset'].unique())

    for i in range(len(datasets)):
        row = '%d & ' % (i + 1)
        row += '%.2f & ' % (df[(df['dataset'] == datasets[i]) & (df['classifier'] == 'RandomForest')]['accuracy'])

        for classifier in ['SVM', 'kNN', 'DecisionTree', 'NaiveBayes']:
            rs_score = df[(df['dataset'] == datasets[i]) &
                          (df['classifier'] == classifier) &
                          (df['method'] == 'RandomSubspace')]['accuracy'].iloc[0]

            row += '%.2f & ' % rs_score

            for alpha in ['0.0', '0.25', '0.5', '0.75', '1.0']:
                ds_score = df[(df['dataset'] == datasets[i]) &
                              (df['classifier'] == classifier) &
                              (df['method'] == 'DeterministicSubspace') &
                              (df['alpha'] == alpha)]['accuracy'].iloc[0]

                if round(ds_score, 2) >= round(rs_score, 2):
                    row += '\cellcolor[gray]{0.8} '

                row += '%.2f & ' % ds_score

        print row[:-2] + ' \\\\'

    print ('\\hline\n'
           '\\end{tabular}\n'
           '}\n'
           '\\end{table}\n')


for k in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
    print_table(k)
