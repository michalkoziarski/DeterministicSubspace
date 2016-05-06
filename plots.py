import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from datasets import *


if len(sys.argv) > 1:
    root_path = sys.argv[1]
else:
    root_path = 'results'


plt.style.use('ggplot')
matplotlib.rc('font', family='Liberation Mono')


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


def load_df_and_datasets(k=None, b=None, root_path='results'):
    frames = []

    for file_name in os.listdir(root_path):
        frames.append(pd.read_csv(os.path.join(root_path, file_name), sep=',', skipinitialspace=True))

    df = pd.concat(frames)

    if k:
        df = df[df['k'] == k].reset_index()

    if b:
        df = df[(df['b'] == b) | (df['b'] == '-')].reset_index()

    datasets = df['dataset'].unique()
    features = get_number_of_features(df)

    return df, datasets[np.argsort(features)]


def mean_accuracy(df, method=None, classifier=None, k=None, alpha=None, b=None):
    selection = df

    if method:
        selection = selection[selection['method'] == method]

    if classifier:
        selection = selection[selection['classifier'] == classifier]

    if k:
        selection = selection[selection['k'] == k]

    if alpha:
        selection = selection[selection['alpha'] == alpha]

    if b:
        selection = selection[selection['b'] == b]

    return np.mean(np.array(filter(lambda x: x != '-', selection['accuracy'])).astype(np.float))


def plot_accuracy_bars(df):
    accuracies = [mean_accuracy(df, classifier='RandomForest')]
    labels = ['RandomForest']
    colors = ['#E24A33']

    for clf in [clf for clf in df['classifier'].unique() if clf != 'RandomForest']:
        accuracies.append(mean_accuracy(df, method='RandomSubspace', classifier=clf))
        labels.append('%s - RS' % clf)
        colors.append('#7A68A6')

        for alpha in [alpha for alpha in df['alpha'].unique() if alpha != '-']:
            accuracies.append(mean_accuracy(df, method='DeterministicSubspace', classifier=clf, alpha=alpha))
            labels.append('%s - DS($\\alpha = %s$)' % (clf, alpha))
            colors.append('#348ABD')

    indices = range(len(accuracies))

    plt.figure()
    plt.bar(map(lambda x: x + 0.1, indices), accuracies, color=colors)
    plt.xticks(map(lambda x: x + 0.5, indices), labels, rotation='vertical')
    plt.xlim((0 - 0.15, len(indices) + 0.15))
    plt.ylim((0., 1.))
    plt.title('Average accuracy over all datasets', y=1.03)
    plt.ylabel('Classification accuracy', labelpad=10)
    plt.savefig('plots_%s_bars.png' % root_path)


def plot_accuracy_k(df):
    for clf in [clf for clf in df['classifier'].unique() if clf != 'RandomForest']:
        k_values = range(5, 55, 5)
        alphas = [alpha for alpha in df['alpha'].unique() if alpha != '-']
        methods = ['RS'] + ['DS(%s)' % alpha for alpha in alphas]

        accuracy = [[] for _ in methods]

        for k in k_values:
            accuracy[0].append(mean_accuracy(df, method='RandomSubspace', k=k, classifier=clf))

            for i in range(len(alphas)):
                alpha = alphas[i]
                accuracy[i + 1].append(mean_accuracy(df, method='DeterministicSubspace', k=k, classifier=clf, alpha=alpha))

        plt.figure()

        for acc in accuracy:
            plt.plot(k_values, acc)

        plt.title('Average accuracy over all datasets for %s' % clf, y=1.03)
        plt.ylabel('Classification accuracy', labelpad=10)
        plt.xlabel('k')
        plt.legend(methods, loc=4)
        plt.savefig('plots_%s_%s.png' % (root_path, clf))


if __name__ == '__main__':
    df, datasets = load_df_and_datasets(root_path=root_path)

    plot_accuracy_bars(df)
    plot_accuracy_k(df)
