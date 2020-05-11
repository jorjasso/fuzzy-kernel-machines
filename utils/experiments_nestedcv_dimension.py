from __future__ import print_function
from __future__ import division

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import sys
import os

qprc_path = os.path.abspath(os.path.join('..'))
if qprc_path not in sys.path:
    sys.path.append(qprc_path)

from utils.plots import *
import pandas as pd
import numpy as np
from kernelfuzzy.fuzzysystem import *
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from fylearn.nfpc import FuzzyPatternClassifier  #base class for fuzzy pattern classifiers (see parameters)
from fylearn.garules import MultimodalEvolutionaryClassifier #learns rules using genetic algorithm
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier #builds fuzzy pattern trees using top-down method.
from fylearn.frr import FuzzyReductionRuleClassifier # based on learning membership functions from min/max.
from fylearn.fpcga import FuzzyPatternClassifierGA # optimizes membership functions globally.
from fylearn.fpt import FuzzyPatternTreeClassifier # builds fuzzy pattern trees using bottom-up method.


def main():
    # Number of random trials

    #PIPELINES

    # SVM +NS kernel
    pipe_NS = Pipeline([
        ('Fuzzifier', NonSingletonFuzzifier(constant_std=True)),
        ('kernel', NonSingletonKernel()),
        ('svm', SVC())])

    #SVM + KBF kernel
    pipe_KBF = Pipeline([
        ('Fuzzifier', NonSingletonFuzzifier(constant_std=True)),
        ('kernel', KBFkernel()),
        ('svm', SVC())])

    #SVM + RBF kernel
    pipe_SVM = Pipeline([('svm', SVC())])

    cv_params_SVM = dict([
        ('svm__gamma', 2.0 ** np.arange(-20, 20)),
        ('svm__C', 2.0 ** np.arange(-15, 15)),
    ])

    #KNN
    pipe_KNN = Pipeline([('knn', KNeighborsClassifier())])

    cv_params_KNN = dict([
        ('knn__n_neighbors', [2, 4, 8, 16, 32])
    ])



    data_dimension = range(2, 100, 1)

    fuzzy_classifiers = (FuzzyPatternClassifier(),
                         MultimodalEvolutionaryClassifier(),
                         FuzzyPatternTreeTopDownClassifier(),
                         FuzzyReductionRuleClassifier(),
                         FuzzyPatternClassifierGA(),
                         FuzzyPatternTreeClassifier())

    data_dimension = range(2, 100, 1)
    n_samples=100
    n_iter = 10
    list_results = []


    for d in data_dimension:
        print("Dimension : {}".format(d))

        # Training and test data
        X, y = make_classification(n_samples=n_samples, n_features=d, n_informative=d, n_redundant=0)

        inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)



        results = {'dimension': d}

        # -------------------------------
        # NSFS trained with  SVM + NS kernel
        # -------------------------------
        cv_params = dict([
            ('Fuzzifier__std_proportion', np.arange(0.01, np.std(X), 0.1)),
            ('kernel__param', 2.0 ** np.arange(-20, 1)),
            ('svm__kernel', ['precomputed']),
            ('svm__C', 2.0 ** np.arange(-15, 15))
        ])
        clf = RandomizedSearchCV(pipe_NS, cv_params, cv=inner_cv, verbose=1, n_jobs=-1, n_iter=n_iter)
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,n_jobs=-1)
        results.update({'NSFS_NS': nested_score.mean(),
                        'NSFS_NS_std': nested_score.std()})

       # results.update(
       #     {'gamma_NS': clf.best_params_['kernel__param'],
       #      'std_fuzzifier_NS': clf.best_params_['Fuzzifier__std_proportion'],
       #      'nro_rules_NS': clf.best_estimator_['svm'].n_support_})

        # ------------------------
        # NSFS trained with SVM + KBF kernel (direct connection)
        # ------------------------

        clf = RandomizedSearchCV(pipe_KBF, cv_params, cv=inner_cv, verbose=1, n_jobs=-1, n_iter=n_iter * 2)
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,n_jobs=-1)
        results.update({'NSFS_KBF': nested_score.mean(),
                        'NSFS_KBF_std': nested_score.std()})


        #results.update(
        #    {'gamma_KBF': clf.best_params_['kernel__param'],
        #     'std_fuzzifier_KBF': clf.best_params_['Fuzzifier__std_proportion'],
        #     'nro_rules_KBF': clf.best_estimator_['svm'].n_support_})

        # C-SVM with Gaussian RBF kernel

        clf = RandomizedSearchCV(pipe_SVM, cv_params_SVM, cv=inner_cv, verbose=1, n_jobs=-1, n_iter=n_iter)
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,n_jobs=-1)
        results.update({'svmRBF': nested_score.mean()})

        # KNN
        clf = RandomizedSearchCV(pipe_KNN, cv_params_KNN, cv=inner_cv, verbose=1, n_jobs=-1, n_iter=n_iter)
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv,n_jobs=-1)
        results.update({'knn': nested_score.mean()})


        # fuzzy classifiers
        for fc in fuzzy_classifiers:
            print('fuzzy classifier ', type(fc).__name__)

            results.update({type(fc).__name__: cross_val_score(fc, X=X, y=y, cv=outer_cv, n_jobs=-1)})

        list_results.append(results)
        # temp saving
        pd.DataFrame(list_results).to_csv('nested_df_dim_' + str(d) + '.csv')

    df_results = pd.DataFrame(list_results)
    df_results.to_csv('nested_df_dimension2_100.csv')


if __name__ == "__main__":
    main()


