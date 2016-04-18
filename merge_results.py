import os
import pandas as pd


frames = []

for file_name in os.listdir('results'):
    frames.append(pd.read_csv(os.path.join('results', file_name), sep=',', skipinitialspace=True))

df = pd.concat(frames).drop('omega', axis=1).reset_index(drop=True)

datasets = df['dataset'].unique()

results = pd.DataFrame(columns=df.columns)


def append(results, dataset, method, classifier, k, alpha, b):
    accuracies = df[(df['dataset'] == dataset) &
                    (df['method'] == method) &
                    (df['classifier'] == classifier) &
                    (df['k'] == k) &
                    (df['alpha'] == alpha) &
                    (df['b'] == b)]['accuracy']

    accuracy = accuracies.mean() if len(accuracies) > 0 else '-'

    results = results.append({
        'dataset': dataset, 'method': method, 'classifier': classifier,
        'k': k, 'alpha': alpha, 'b': b, 'accuracy': accuracy
    }, ignore_index=True)

    return results


for dataset in datasets:
    for k in range(5, 55, 5):
        results = append(results, dataset, '-', 'RandomForest', k, '-', '-')

        for classifier in ['kNN', 'DecisionTree', 'SVM', 'NaiveBayes']:
            results = append(results, dataset, 'RandomSubspace', classifier, k, '-', '-')

            for alpha in ['0.0', '0.25', '0.5', '0.75', '1.0']:
                for b in ['1', '2', '3']:
                    results = append(results, dataset, 'DeterministicSubspace', classifier, k, alpha, b)

results.to_csv('merged.csv')
