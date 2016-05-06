from subspace import *
from precalculate_mi import *
from time import gmtime, strftime
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold


if len(sys.argv) > 1:
    try:
        dataset = globals()['load_' + sys.argv[1]]()
    except:
        dataset = globals()['load_keel'](sys.argv[1])

    datasets = {sys.argv[1]: dataset}
else:
    datasets = load_all()

if len(sys.argv) > 2:
    K = [int(sys.argv[2])]
else:
    K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

if len(sys.argv) > 3:
    classifiers = {sys.argv[3]: eval(sys.argv[3])}
else:
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'kNN': KNeighborsClassifier(),
        'SVM': LinearSVC(),
        'NaiveBayes': GaussianNB()
    }
    
if len(sys.argv) > 4:
    root_path = sys.argv[4]
else:
    root_path = 'results'
    
if len(sys.argv) > 5:
    tested_classifier = eval(sys.argv[5])
else:
    tested_classifier = DeterministicSubspaceClassifier


def test(X, y, train_idx, test_idx, clf, dataset_name, clf_name, k, method='-', alpha='-', date=None):
    file_name = '%s_%s_k_%d_%s.csv' % (dataset_name, clf_name, k, date)

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    if not os.path.exists(os.path.join(root_path, file_name)):
        with open(os.path.join(root_path, file_name), 'w') as f:
            f.write('dataset, method, classifier, k, alpha, accuracy\n')

    score = clone(clf).fit(X[train_idx], y[train_idx]).score(X[test_idx], y[test_idx])

    with open(os.path.join(root_path, file_name), 'a') as f:
        f.write('%s, %s, %s, %d, %s, %.3f\n' % (dataset_name, method, clf_name, k, alpha, score))

    print 'Dataset: %s, method: %s, classifier: %s, k: %d, alpha: %s, accuracy: %.3f' % \
          (dataset_name, method, clf_name, k, alpha, score)


for dataset_name, dataset in datasets.iteritems():
    date = strftime('%Y_%m_%d_%H-%M-%S', gmtime())
    X, y = dataset
    n = X.shape[1] / 2

    for train_idx, test_idx in KFold(len(y), n_folds=2, shuffle=True):
        for k in K:
            test(X, y, train_idx, test_idx, RandomForestClassifier(n_estimators=k), dataset_name,
                 'RandomForest', k, date=date)

            for clf_name, clf in classifiers.iteritems():
                test(X, y, train_idx, test_idx, RandomSubspaceClassifier(clf, k=k, n=n), dataset_name,
                     clf_name, k, 'RandomSubspace', date=date)

                for alpha in [0., .1, .2, .3, .4, .5, .6, .7, .8, .9]:
                    test(X, y, train_idx, test_idx, tested_classifier(clf, k=k, n=n, alpha=alpha),
                         dataset_name, clf_name, k, 'DeterministicSubspace', alpha, date=date)
