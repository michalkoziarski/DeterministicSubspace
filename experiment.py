from trial import run_trial
from datasets import get_names


CLASSIFIERS = ["CART", "kNN", "SVM", "NaiveBayes", "ParzenKDE", "NNKDE", "GMM"]
MEASURES = ["accuracy", "mutual_information", "correlation"]
DATASETS = get_names()


def prepare_args(dataset, fold, classifier, method='-', measure='-', k='-', n='-', alpha='-', repeat=False):
    if alpha != '-':
        alpha = str(round(alpha, 1))

    return {
        'dataset': dataset,
        'fold': str(fold),
        'classifier': classifier,
        'method': method,
        'measure': measure,
        'k': str(k),
        'n': str(n),
        'alpha': alpha,
        'repeat': repeat
    }

for dataset in DATASETS:
    for fold in range(10):
        for k in range(5, 55, 5):
            print 'Running trial -dataset %s -fold %s -classifier RandomForest -k %s...' % (dataset, fold, k)
            run_trial(prepare_args(dataset, fold, 'RandomForest', k=k))

            for classifier in CLASSIFIERS:
                print 'Running trial -dataset %s -fold %s -classifier %s -method RS -k %s...' % \
                      (dataset, fold, classifier, k)
                run_trial(prepare_args(dataset, fold, classifier, method='RS', k=k))

                for alpha in [i / 10. for i in range(11)]:
                    for measure in MEASURES:
                        print 'Running trial -dataset %s -fold %s -classifier %s -method DS -k %s -measure %s -alpha %s...' % \
                              (dataset, fold, classifier, k, measure, alpha)
                        run_trial(prepare_args(dataset, fold, classifier, method='DS', k=k, measure=measure, alpha=alpha))
