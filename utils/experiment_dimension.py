from __future__ import print_function
from __future__ import division


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
    n_iter = 10
    list_results = []
    for d in data_dimension:
        print("Dimension : {}".format(d))

        #Training and test data
        X_, y_ = make_classification(n_samples=500, n_features=d, n_informative=d, n_redundant=0)

        train_samples = 250  # Samples used for training the models

        X = X_[:train_samples]
        y = y_[:train_samples]

        X_test = X_[train_samples:]
        y_test = y_[train_samples:]

        results = {'dimension': d}

        #-------------------------------
        # NSFS trained with  SVM + NS kernel
        #-------------------------------
        cv_params = dict([
            ('Fuzzifier__std_proportion', np.arange(0.01, np.std(X), 0.1)),
            ('kernel__param', 2.0 ** np.arange(-20, 1)),
            ('svm__kernel', ['precomputed']),
            ('svm__C', 2.0 ** np.arange(-15, 15))
        ])
        clf = RandomizedSearchCV(pipe_NS, cv_params, cv=5, verbose=1, n_jobs=-1, n_iter=n_iter)
        clf.fit(X, y)
        results.update({'NSFS_NS': accuracy_score(y_test, clf.predict(X_test))})


        # predict
        # for opt in list_options_predict:
        #    K,y_pred=NSFS_predict(clf, X,X_test, option=opt)
        #    if opt==0 or opt==1 or opt==4:
        #        print("acc K", accuracy_score(y_test, clf.best_estimator_['svm'].predict(K)))
        #    print("acc y)", accuracy_score(y_test, sign_fun(y_pred)))

        #----------------------------------------
        # NSFS trained with SVM+NS then predict with  with KBF kernel, i.e., add the denominator
        #----------------------------------------
        K, y_pred = NSFS_predict(clf, X, X_test, option=0)
        results.update({'NSFS_NS_KBF': accuracy_score(y_test, clf.best_estimator_['svm'].predict(K))})

        results.update(
            {'gamma_NS': clf.best_params_['kernel__param'],
             'std_fuzzifier_NS': clf.best_params_['Fuzzifier__std_proportion'],
             'nro_rules_NS': clf.best_estimator_['svm'].n_support_})

        #------------------------
        # NSFS trained with SVM + KBF kernel (direct connection)
        #------------------------

        clf = RandomizedSearchCV(pipe_KBF, cv_params, cv=5, verbose=1, n_jobs=-1, n_iter=n_iter*2)
        clf.fit(X, y)
        results.update({'NSFS_KBF': accuracy_score(y_test, clf.predict(X_test))})

        results.update(
            {'gamma_KBF': clf.best_params_['kernel__param'],
             'std_fuzzifier_KBF': clf.best_params_['Fuzzifier__std_proportion'],
             'nro_rules_KBF': clf.best_estimator_['svm'].n_support_})

        # C-SVM with Gaussian RBF kernel

        model_SVM = RandomizedSearchCV(pipe_SVM, cv_params_SVM, cv=5, verbose=1, n_jobs=-1, n_iter=n_iter)
        model_SVM.fit(X, y)

        # KNN
        neigh = RandomizedSearchCV(pipe_KNN, cv_params_KNN, cv=5, verbose=1, n_jobs=-1, n_iter=n_iter)
        neigh.fit(X, y)

        results.update({'svmRBF': model_SVM.score(X_test, y_test),
                        'knn': neigh.score(X_test, y_test)})

        # fuzzy classifiers
        for fc in fuzzy_classifiers:
            print('fuzzy classifier ', type(fc).__name__)
            results.update({type(fc).__name__: accuracy_score(y_test, fc.fit(X, y).predict(X_test))})

        list_results.append(results)
        # temp saving
        pd.DataFrame(list_results).to_csv('temp_df_dim_' + str(d) + '.csv')

    df_results = pd.DataFrame(list_results)
    df_results.to_csv('df_dimension2_100.csv')


if __name__ == "__main__":
    main()



