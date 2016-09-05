## Forming classifier ensembles with deterministic feature subspaces

Python code for forming classifier ensembles with deterministic feature subspace approach. Contains deterministic subspace classifier compatible with scikit-learn interface and tools necessary to easily repeat conducted experiments.

More details about the method, results of experiments and related papers can be found at [mkoziarski.com/deterministic-feature-subspace-method](https://mkoziarski.com/deterministic-feature-subspace-method).

## Usage

#### Requirements

Tested on Python 2.7.9. Remaining packages used are enclosed in [requirements.txt](https://github.com/michalkoziarski/DeterministicSubspace/blob/master/requirements.txt).

#### Initialization

To download necessary datasets and create databases in which results will be stored go to main project directory and execute

`python initialize.py`

Results reported in last paper were obtained on precalculated folds, enclosed in this repository. If you want to evaluate different set of folds, you can run

(optional) `python precalculate_folds.py`

#### Scheduling trials

Experiment was designed to be run from several processes at once. Because of that queue of pending trials has to be filled first.

By default, all trials described in the last paper will be run. If you want to change that, you can modify [schedule_experiment.py](https://github.com/michalkoziarski/DeterministicSubspace/blob/master/schedule_experiment.py) accordingly. After that, execute

`python schedule_experiment.py`

#### Running experiment

Main script will try to pull pending trials and evaluate them as long as they are present.

`python experiment.py`

#### Exporting results to CSV file

After the experiment is done, you can convert results to CSV format by running

`python -c 'import database; database.export()'`
