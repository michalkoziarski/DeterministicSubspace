import sys
import argparse
import cPickle as pickle
import database as db

from subspace import *
from datasets import quick_load, get_names
from nonparametric import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


AVAILABLE_DATASETS = get_names()
AVAILABLE_CLASSIFIERS = {'CART': DecisionTreeClassifier, 'kNN': KNeighborsClassifier, 'SVM': LinearSVC,
                         'NaiveBayes': GaussianNB, 'ParzenKDE': ParzenKernelDensityClassifier,
                         'NNKDE': NNKernelDensityClassifier, 'GMM': GMMClassifier,
                         'RandomForest': RandomForestClassifier}


def run_trial(args):
    X, y = quick_load(args['dataset'])

    if args['n'] == '-':
        n = X.shape[1] / 2
    else:
        n = args['n']

    if args['k'] == '-':
        if args['method'] in ['DS', 'RS']:
            raise AttributeError('Argument "k" required for DS and RS methods.')

        k = args['k']
    else:
        k = int(args['k'])

    if args['alpha'] == '-':
        if args['method'] == 'DS':
            raise AttributeError('Argument "alpha" required for DS method.')
        alpha = args['alpha']
    else:
        alpha = float(args['alpha'])

    fold = int(args['fold'])

    if args['classifier'] == 'RandomForest':
        base_classifier = RandomForestClassifier(n_estimators=k)
    else:
        base_classifier = AVAILABLE_CLASSIFIERS[args['classifier']]()

    if args['method'] == 'DS':
        if args['measure'] == 'accuracy':
            method = DeterministicSubspaceClassifier
        elif args['measure'] == 'mutual_information':
            method = MIDeterministicSubspaceClassifier
        elif args['measure'] == 'correlation':
            method = CorrDeterministicSubspaceClassifier
        else:
            raise AttributeError('Proper quality measure of DS method has to be specified.')

        classifier = method(base_classifier, k=k, n=n, alpha=alpha)
    elif args['method'] == 'RS':
        classifier = RandomSubspaceClassifier(base_classifier, k=k, n=n)
    else:
        classifier = base_classifier

    arguments = (args['dataset'], args['fold'], args['classifier'], args['method'], args['measure'], args['k'],
                 args['n'], args['alpha'])

    if args['repeat'] is False and not db.reserve(*arguments):
        print 'Not repeating the trial, quiting.'

        return

    folds = pickle.load(open('folds.pickle', 'r'))

    train_idx, test_idx = folds[args['dataset']][fold]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print 'Calculating test error...'

    score = classifier.fit(X_train, y_train).score(X_test, y_test)

    print 'Test error of %.4f calculated.' % score
    print 'Trying to save the result to database...'

    if args['repeat']:
        db.insert(*arguments, score=score)
    else:
        db.update(*arguments, score=score)

    print 'Results saved.'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate test error for single fold and store it in database.')

    parser.add_argument('-dataset', help='dataset name', choices=AVAILABLE_DATASETS, required=True)
    parser.add_argument('-fold', type=int, help='index of precalculated 5x2 cross-validation fold', choices=range(0, 10),
                        required=True)
    parser.add_argument('-classifier', help='base classifier name', choices=AVAILABLE_CLASSIFIERS.keys(), required=True)
    parser.add_argument('-method', help='subspace method, either deterministic (DS), random (RS), or none (-, default)',
                        choices=['DS', 'RS', '-'], default='-')
    parser.add_argument('-measure', help='quality measure of DS method',
                        choices=['accuracy', 'mutual_information', 'correlation', '-'], default='-')
    parser.add_argument('-k', help='number of subspaces', default='-')
    parser.add_argument('-n', help='number of features per subspace, half of total by default', default='-')
    parser.add_argument('-alpha', help='quality coefficient of DS method, value in range from 0 to 1', default='-')
    parser.add_argument('-repeat', type=bool, help='whether to conduct experiment again if result is already present',
                        default=False)

    args = vars(parser.parse_args())

    run_trial(args)
