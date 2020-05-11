import warnings
warnings.filterwarnings('ignore')
import sys
import os
qprc_path = os.path.abspath(os.path.join('..'))
if qprc_path not in sys.path:
    sys.path.append(qprc_path)

#from utils.plots import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import numpy.ma as ma
import csv
from scipy.stats import randint
from imblearn.over_sampling import SMOTE


import seaborn as sns
from kernelfuzzy.kernels import KBFkernelSymmetric
from  kernelfuzzy.fuzzyset import FuzzySet
from kernelfuzzy.fuzzysystem import *
from sklearn.datasets.samples_generator import make_classification
from sklearn.datasets import make_moons, make_circles,make_blobs,load_digits
from sklearn.svm import SVC,NuSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz # for FCM
import pickle
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from c45 import C45
from sklearn.preprocessing import StandardScaler, MinMaxScaler,RobustScaler, Normalizer,QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from fylearn.nfpc import FuzzyPatternClassifier
from fylearn.garules import MultimodalEvolutionaryClassifier
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier

from fylearn.nfpc import FuzzyPatternClassifier  #base class for fuzzy pattern classifiers (see parameters)
from fylearn.garules import MultimodalEvolutionaryClassifier #learns rules using genetic algorithm
from fylearn.fpt import FuzzyPatternTreeTopDownClassifier #builds fuzzy pattern trees using top-down method.
from fylearn.frr import FuzzyReductionRuleClassifier # based on learning membership functions from min/max.
from fylearn.fpcga import FuzzyPatternClassifierGA # optimizes membership functions globally.
from fylearn.fpt import FuzzyPatternTreeClassifier # builds fuzzy pattern trees using bottom-up method.


def plot_class_distribution(y, str_title):
    unique, counts = np.unique(y, return_counts=True)
    plt.bar(unique, counts)
    plt.title(str_title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()


def get_Keel_data(keel_path):
    '''
    get all keel data (*.dat files) within a folder,
    create a list of dataframes,
    concatenate the dataframes,
    perform the intersection of the rows

    '''
    l_df = []
    for file in os.listdir(keel_path):
        if file.endswith(".dat"):
            print(os.path.join(keel_path, file))
            l_df.append(get_Keel_data_file(os.path.join(keel_path, file)))

    df = pd.concat(l_df)
    print('before eliminating duplicates  shape', df.shape)
    df = df.drop_duplicates().reset_index(drop=True)
    print('after eliminating duplicates  shape', df.shape)
    # generating labels for predictors and target
    column_names = ['x' + str(i) for i in range(df.shape[1] - 1)]
    df.columns = column_names + ['y']

    X = df.drop('y', axis=1).to_numpy().astype(float)
    y = df['y'].to_numpy()
    return X, y, df


def get_Keel_data_file(keel_path):
    l = list()
    with open(keel_path) as fp:
        for cnt, line in enumerate(fp):
            if not line.startswith('@'):
                s = line.strip().split(',')
                ll = [e for e in s if e.strip()]
                l.append(ll)
        df = pd.DataFrame.from_records(l)

        return df


def classify(dataset_name, classifier, cv_params, pipe, n_iter, inner_cv, outer_cv, X, y, noise):
    print(classifier, ', noise = ', noise)
    clf = RandomizedSearchCV(pipe, cv_params, cv=inner_cv, verbose=0, n_jobs=-1, n_iter=n_iter)
    l_results = []

    # with joblib.parallel_backend('dask'):

    start = time.time()
    nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv, n_jobs=-1)
    end = time.time()
    elapsed_time = end - start
    print(nested_score)
    n_trials = outer_cv.get_n_splits()
    for trial in range(0, n_trials):
        results = {'trial': trial, 'noise': noise, 'exec_time': elapsed_time, 'data': dataset_name}
        results.update({classifier: nested_score[trial]})
        l_results.append(results)
    return l_results


def do_experiments(experiment_description):
    def warn(*args, **kwargs):
        pass

    import warnings
    warnings.warn = warn
    # run block of code and catch warnings
    with warnings.catch_warnings():
        # ignore all caught warnings
        warnings.filterwarnings("ignore")
        # execute code that will generate warnings
        # EXPERIMENTAL SETUP
    n_iter = experiment_description['n_iter']
    fileNameExperiments = experiment_description['output_dir']
    n_splits_outter = experiment_description['n_splits_outter']
    n_splits_inner = experiment_description['n_splits_inner']
    classifier = experiment_description['classifier']
    output_dir = experiment_description['output_dir']
    type_noise = experiment_description['type_noise']  # options are nn, nc, cn
    dataset_name = experiment_description['dataset_name']
    noise_level = experiment_description['noise_level']

    pd.DataFrame(experiment_description).to_csv(output_dir + '/experiment_description.csv')
    list_scalers = [MinMaxScaler(), None]
    scaler = MinMaxScaler()

    # PIPELINES
    # C4.5
    cv_params_c45 = {'scaler': list_scalers}
    # BagC4.5
    cv_params_bagc45 = {'scaler': list_scalers,
                        'model__n_estimators': [1, 10, 50]}
    pipe_bagc45 = Pipeline([('scaler', scaler),
                            ('model', BaggingClassifier(base_estimator=C45(), n_jobs=-1))])

    # naybe bayes
    # Cart
    cv_params_cart = {'scaler': list_scalers, 'model__max_depth': [2, 4, 8, 16]}
    pipe_cart = Pipeline([('scaler', scaler),
                          ('model', DecisionTreeClassifier())])
    # BagCart

    # ------------------------
    # logistic regresion
    cv_params_lr = {'scaler': list_scalers, 'model__C': np.logspace(-2, 3, 6), 'model__penalty': ['l1', 'l2']}
    pipe_lr = Pipeline([('scaler', scaler), ('model', LogisticRegression(solver='liblinear'))])

    # Random forest
    cv_params_rf = {'scaler': list_scalers,
                    "model__max_depth": [10, None],
                    "model__max_features": randint(1, 5),
                    "model__min_samples_split": randint(2, 15),
                    "model__criterion": ["gini", "entropy"],
                    "model__min_samples_leaf": randint(1, 15),
                    "model__bootstrap": [True, False]}

    pipe_rf = Pipeline([('scaler', scaler), ('model', RandomForestClassifier())])

    # MLP
    cv_params_mlp = {'scaler': list_scalers,
                     'model__alpha': np.logspace(-2, 3, 6)}
    pipe_mlp = Pipeline([('scaler', scaler), ('model', MLPClassifier())])

    # sgd
    cv_params_sgd = {'scaler': list_scalers,
                     'model__alpha': np.logspace(-2, 3, 6),
                     'model__penalty': ['l1', 'l2', 'elasticnet']}
    pipe_sgd = Pipeline([('scaler', scaler), ('model', SGDClassifier())])

    # SVM +NS kernel
    pipe_NS = Pipeline([('scaler', scaler),
                        ('Fuzzifier', NonSingletonFuzzifier(constant_std=True)),
                        ('kernel', NonSingletonKernel()),
                        ('svm', SVC())])

    # SVM + KBF kernel
    pipe_KBF = Pipeline([('scaler', scaler),
                         ('Fuzzifier', NonSingletonFuzzifier(constant_std=True)),
                         ('kernel', KBFkernel()),
                         ('svm', SVC())])

    # SVM + symmetric KBF kernel
    pipe_KBF_symmetric = Pipeline([('scaler', scaler),
                                   ('Fuzzifier', NonSingletonFuzzifier(constant_std=True)),
                                   ('kernel', KBFkernelSymmetric()),
                                   ('svm', SVC())])

    # SVM + RBF kernel
    pipe_SVM = Pipeline([('scaler', scaler), ('svm', SVC())])

    cv_params_SVM = dict([('scaler', list_scalers),
                          ('svm__gamma', 2.0 ** np.arange(-20, 20)),
                          ('svm__C', 2.0 ** np.arange(-15, 15))])

    # KNN
    pipe_KNN = Pipeline([('scaler', scaler),
                         ('knn', KNeighborsClassifier())])

    cv_params_KNN = dict([('scaler', list_scalers),
                          ('knn__n_neighbors', [2, 4, 8])])

    fuzzy_classifiers = (FuzzyPatternClassifier(),
                         MultimodalEvolutionaryClassifier(),
                         FuzzyPatternTreeTopDownClassifier(),
                         FuzzyReductionRuleClassifier(),
                         FuzzyPatternClassifierGA(),
                         FuzzyPatternTreeClassifier())

    list_results = []

    prefix_path = '../data/attributenoise/' + type_noise + '/attribute_noise_'

    for noise in noise_level:
        ################################
        # READING DATASETS
        ################################

        temp_name = dataset_name + '-' + str(noise) + 'an-' + type_noise
        noisy_data_foldername = prefix_path + str(noise) + '_' + type_noise + '/' + temp_name + '/'
        X, y, df = get_Keel_data(noisy_data_foldername)

        #if noise == 5: plot_class_distribution(y, 'Class Frequency before Filtering')
        # print(df.head())
        ##############################################
        # FILTERING DATA WITH LESS THAN 25 OBSERVATIONS
        ##############################################
        unique_elements, counts_elements = np.unique(y, return_counts=True)
        filter_labels = unique_elements[counts_elements < 20]
        # filtering
        df = df[~df.y.isin(filter_labels)]
        # update
        X = df.drop('y', axis=1).to_numpy().astype(float)
        y = df['y'].to_numpy()

        ##############################################
        # DEALING WITH INBALANCED DATA WITH SMOTE
        #############################################
        # class distribution
        #if noise == 5: plot_class_distribution(y, 'Class Frequency before SMOTE')
        # oversampling with SMOTE
        sm = SMOTE()
        X, y = sm.fit_sample(X, y)
        #if noise == 5: plot_class_distribution(y, 'Class Frequency before SMOTE')

        inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=0)
        outer_cv = StratifiedKFold(n_splits=n_splits_outter, shuffle=True, random_state=0)

        std_val_X = np.std(X)
        std_val_MinMax = np.std(MinMaxScaler().fit(X).transform(X))
        cv_params = [dict([('scaler', [MinMaxScaler()]),
                           ('Fuzzifier__std_proportion', np.arange(0.01, std_val_MinMax, std_val_MinMax / 20)),
                           ('kernel__param', 2.0 ** np.arange(-20, 1)),
                           ('svm__kernel', ['precomputed']),
                           ('svm__C', 2.0 ** np.arange(-15, 15))]),
                     dict([('scaler', [None]),
                           ('Fuzzifier__std_proportion', np.arange(0.01, std_val_X, std_val_X / 20)),
                           ('kernel__param', 2.0 ** np.arange(-20, 1)),
                           ('svm__kernel', ['precomputed']),
                           ('svm__C', 2.0 ** np.arange(-15, 15))])
                     ]

        # ------------------------------------
        # NSFS trained with  SVM + NS kernel
        # -------------------------------
        if classifier == 'NSFS_NS':
            list_results = list_results + classify(dataset_name, classifier, cv_params, pipe_NS, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)
            # cumulative saving
            # filename=output_dir+'/'+'cum_dim'+ str(noise)+'_'+classifier+'.csv'
            # pd.DataFrame(list_results).to_csv(filename)

        # ------------------------
        # NSFS trained with SVM + KBF kernel (direct connection) using only the upper traingular part
        # ------------------------
        if classifier == 'NSFS_KBF':
            list_results = list_results + classify(dataset_name, classifier, cv_params, pipe_KBF, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        # ------------------------
        # NSFS trained with SVM + KBF symmetric kernel (direct connection)
        # ------------------------
        if classifier == 'NSFS_KBF_symmetric':
            list_results = list_results + classify(dataset_name, classifier, cv_params, pipe_KBF_symmetric, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        # -------------------------------------------------------------------------------------
        if classifier == 'svmRBF':
            list_results = list_results + classify(dataset_name, classifier, cv_params_SVM, pipe_SVM, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        if classifier == 'knn':
            list_results = list_results + classify(dataset_name, classifier, cv_params_KNN, pipe_KNN, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        if classifier == 'lr':
            list_results = list_results + classify(dataset_name, classifier, cv_params_lr, pipe_lr, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        if classifier == 'rf':
            list_results = list_results + classify(dataset_name, classifier, cv_params_rf, pipe_rf, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        if classifier == 'mlp':
            list_results = list_results + classify(dataset_name, classifier, cv_params_mlp, pipe_mlp, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        if classifier == 'sgd':
            list_results = list_results + classify(dataset_name, classifier, cv_params_sgd, pipe_sgd, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

        if classifier == 'bagC45':
            list_results = list_results + classify(dataset_name, classifier, cv_params_bagc45, pipe_bagc45, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        # ---------------
        cv_params_fuzz = {'scaler': list_scalers}

        if classifier == 'FuzzyPatternClassifier':
            pipe_fuzz = Pipeline([('scaler', scaler),
                                  ('model', fuzzy_classifiers[0])])
            list_results = list_results + classify(dataset_name, classifier, cv_params_fuzz, pipe_fuzz, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        if classifier == 'MultimodalEvolutionaryClassifier':
            pipe_fuzz = Pipeline([('scaler', scaler),
                                  ('model', fuzzy_classifiers[1])])
            list_results = list_results + classify(dataset_name, classifier, cv_params_fuzz, pipe_fuzz, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        if classifier == 'FuzzyPatternTreeTopDownClassifier':
            pipe_fuzz = Pipeline([('scaler', scaler),
                                  ('model', fuzzy_classifiers[2])])
            list_results = list_results + classify(dataset_name, classifier, cv_params_fuzz, pipe_fuzz, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        if classifier == 'FuzzyReductionRuleClassifier':
            pipe_fuzz = Pipeline([('scaler', scaler),
                                  ('model', fuzzy_classifiers[3])])
            list_results = list_results + classify(dataset_name, classifier, cv_params_fuzz, pipe_fuzz, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        if classifier == 'FuzzyPatternClassifierGA':
            pipe_fuzz = Pipeline([('scaler', scaler),
                                  ('model', fuzzy_classifiers[4])])
            list_results = list_results + classify(dataset_name, classifier, cv_params_fuzz, pipe_fuzz, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        if classifier == 'FuzzyPatternTreeClassifier':
            pipe_fuzz = Pipeline([('scaler', scaler),
                                  ('model', fuzzy_classifiers[5])])
            list_results = list_results + classify(dataset_name, classifier, cv_params_fuzz, pipe_fuzz, n_iter,
                                                   inner_cv, outer_cv, X, y, noise)

        if classifier == 'C4.5':
            pipe_c45 = Pipeline([('scaler', scaler), ('model', C45())])
            list_results = list_results + classify(dataset_name, classifier, cv_params_c45, pipe_c45, n_iter, inner_cv,
                                                   outer_cv, X, y, noise)

            # Saving results
    filename = output_dir + '/' + type_noise + '_' + dataset_name + '_' + classifier + '.csv'
    pd.DataFrame(list_results).to_csv(filename)
