import database as db
import datasets


CLASSIFIERS = ["CART", "kNN", "SVM", "NaiveBayes", "ParzenKDE", "NNKDE", "GMM"]
MEASURES = ["accuracy", "mutual_information", "correlation"]
KS = range(5, 55, 5)
ALPHAS = [i / 10. for i in range(11)]

FOLDS = range(10)
DATASETS = datasets.get_names()


def prepare_args(dataset, fold, classifier, method='-', measure='-', k='-', n='-', alpha='-'):
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
        'alpha': alpha
    }


for dataset in DATASETS:
    for fold in FOLDS:
        for k in KS:
            db.add_to_pending(prepare_args(dataset, fold, 'RandomForest', k=k))

            for classifier in CLASSIFIERS:
                db.add_to_pending(prepare_args(dataset, fold, classifier, method='RS', k=k))

                for alpha in ALPHAS:
                    for measure in MEASURES:
                        db.add_to_pending(prepare_args(dataset, fold, classifier, method='DS', k=k,
                                                       measure=measure, alpha=alpha))
