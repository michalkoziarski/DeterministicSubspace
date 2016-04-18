import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from latex import load_df_and_datasets


plt.style.use('ggplot')
matplotlib.rc('font', family='Liberation Mono')


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

    for clf in ['SVM', 'kNN', 'NaiveBayes', 'DecisionTree']:
        accuracies.append(mean_accuracy(df, method='RandomSubspace', classifier=clf))
        labels.append('%s - RS' % clf)
        colors.append('#7A68A6')

        for b in ['1', '2', '3']:
            for alpha in ['0.0', '0.25', '0.5', '0.75', '1.0']:
                accuracies.append(mean_accuracy(df, method='DeterministicSubspace', classifier=clf, alpha=alpha, b=b))
                labels.append('%s - DS($\\alpha = %s, b = %s$)' % (clf, alpha, b))
                colors.append('#348ABD')

    indices = range(len(accuracies))

    plt.figure()
    plt.bar(map(lambda x: x + 0.1, indices), accuracies, color=colors)
    plt.xticks(map(lambda x: x + 0.5, indices), labels, rotation='vertical')
    plt.xlim((0 - 0.15, len(indices) + 0.15))
    plt.ylim((0., 1.))
    plt.title('Average accuracy over all datasets', y=1.03)
    plt.ylabel('Classification accuracy', labelpad=10)
    plt.show()
    plt.savefig('accuracy_bars.png')


def plot_accuracy_k(df):
    for clf in ['SVM', 'kNN', 'NaiveBayes', 'DecisionTree']:
        k_values = range(5, 55, 5)
        methods = ['RS',
                  'DS(0.0)',
                  'DS(0.25)',
                  'DS(0.5)',
                  'DS(0.75)',
                  'DS(1.0)']
        alphas = ['0.0', '0.25', '0.5', '0.75', '1.0']

        accuracy = [[] for _ in methods]

        for k in k_values:
            accuracy[0].append(df[(df['method'] == 'RandomSubspace') & (df['k'] == k) & (df['classifier'] == clf)]['accuracy'].mean())

            for i in range(len(alphas)):
                alpha = alphas[i]
                accuracy[i + 1].append(df[(df['method'] == 'DeterministicSubspace') & (df['alpha'] == alpha) & (df['k'] == k) & (df['classifier'] == clf)]['accuracy'].mean())

        plt.figure()

        for acc in accuracy:
            plt.plot(k_values, acc)

        plt.title('Average accuracy over all datasets for %s' % clf, y=1.03)
        plt.ylabel('Classification accuracy', labelpad=10)
        plt.xlabel('k')
        plt.legend(methods, loc=4)
        plt.show()
        plt.savefig('accuracy_k_%s.png' % clf)


if __name__ == '__main__':
    df, datasets = load_df_and_datasets()

    plot_accuracy_bars(df)
    plot_accuracy_k(df)
