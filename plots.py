import matplotlib
import matplotlib.pyplot as plt

from latex import load_df_and_datasets


plt.style.use('ggplot')
matplotlib.rc('font', family='Liberation Mono')


def plot_accuracy_bars(df):
    for clf in ['SVM', 'kNN', 'NaiveBayes', 'DecisionTree']:
        accuracy = []
        accuracy.append(df[(df['method'] == 'RandomSubspace') & (df['classifier'] == clf)]['accuracy'].mean())

        for alpha in ['0.0', '0.25', '0.5', '0.75', '1.0']:
            accuracy.append(df[(df['method'] == 'DeterministicSubspace') & (df['classifier'] == clf) & (df['alpha'] == alpha)]['accuracy'].mean())

        indices = range(len(accuracy))
        labels = ['RS',
                  'DS(0.0)',
                  'DS(0.25)',
                  'DS(0.5)',
                  'DS(0.75)',
                  'DS(1.0)']

        plt.figure()
        plt.bar(map(lambda x: x + 0.1, indices), accuracy)
        plt.xticks(map(lambda x: x + 0.5, indices), labels)
        plt.xlim((0 - 0.15, len(indices) + 0.15))
        plt.ylim((0., 1.))
        plt.title('Average accuracy over all datasets for %s' % clf, y=1.03)
        plt.ylabel('Classification accuracy', labelpad=10)
        plt.xlabel('Subspace generation method: random (RS) and deterministic (DS)')

        plt.savefig('accuracy_bars_%s.png' % clf)


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

        plt.savefig('accuracy_k_%s.png' % clf)


if __name__ == '__main__':
    df, datasets = load_df_and_datasets()

    plot_accuracy_bars(df)
    plot_accuracy_k(df)
