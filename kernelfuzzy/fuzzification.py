"""

    Class FuzzyData

"""


import numpy as np
from kernelfuzzy.fuzzyset import FuzzySet
from kernelfuzzy.memberships import gaussmf
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



###
###

class FuzzyData:
    _data = None
    _fuzzydata = None
    _epistemic_values = None  # for epistemic fuzzification
    _std_values = None  # for nonsingleton fuzzification
    _target = None

    def __init__(self, data: pd.DataFrame = None, target: str = None):
        if data is not None:
            self._data = data
            self._target = target

    def quantile_fuzzification_classification(self):

        '''
        
        Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01438607/document

        '''

        grouped = self._data.groupby([self._target])

        self._epistemic_values = grouped.transform(lambda x:
                                                   np.exp(-np.square(x - x.quantile(0.5))
                                                          /
                                                          (np.abs(x.quantile(0.75) - x.quantile(0.25)) / (
                                                                  2 * np.sqrt(2 * np.log(2)))) ** 2
                                                          ))

        # join data and epistemistic values
        num_rows = self._epistemic_values.shape[0]
        num_cols = self._epistemic_values.shape[1]

        self._fuzzydata = np.asarray([[FuzzySet(elements=self._data.iloc[j, i],
                                                membership_degrees=self._epistemic_values.iloc[j, i])
                                       for i in range(num_cols)]
                                      for j in range(num_rows)])

    def get_fuzzydata(self):
        return self._fuzzydata

    def get_data(self):
        return self._data

    def get_epistemic_values(self):
        return self._epistemic_values

    def get_std_values(self):
        return self._std_values


    def get_target(self):
        return self._data[self._target]

    def show_class(self):

        """

        Print in the stdout the all the contents of the class, for debugging

        """

        print("(_data)             \n", _data, "\n")
        print("(_fuzzydata)        \n", _fuzzydata, "\n")
        print("(_epistemic_values) \n", _epistemic_values, "\n")
        print("(_target)           \n", _target, "\n")

    def non_singleton_fuzzification_classification(self, constant_std=True, std_proportion=5):


        grouped = self._data.groupby([self._target])

        if constant_std:
            self._std_values = grouped.transform(lambda x: std_proportion)
        else:

            self._std_values = grouped.transform(lambda x: np.random.normal(np.random.uniform(low=np.std(x)/10, high=np.std(x)/2, size=len(x)),
                                                                            (std_proportion / 100) * np.std(x), len(x)))

        num_rows = self._std_values.shape[0]
        num_cols = self._std_values.shape[1]

        self._fuzzydata = np.asarray([[FuzzySet(membership_function_params=[self._data.iloc[j, i],
                                                                            self._std_values.iloc[j, i]])
                                       for i in range(num_cols)]
                                      for j in range(num_rows)])

        # grouped = self._data.groupby([self._target]).agg(['std'])

    # TOYS DATASETS
    @staticmethod
    def create_toy_fuzzy_dataset(num_rows=10, num_cols=2, parametric=False):

        '''
        
        Creates a matrix of fuzzy datasets, each row represent a tuple of fuzzy sets
        each column is a variable. Each fuzzy set is a fuzzy set with gaussian membership function
        
        '''
        # returns, elements, membership degrees based on gaussmf
        if not parametric:
            return np.asarray([[FuzzySet(elements=np.random.uniform(0, 100, 2),
                                         membership_function=gaussmf,
                                         membership_function_params=[np.mean(np.random.uniform(0, 100, 2)),
                                                                     np.std(np.random.uniform(0, 100, 2))])
                                for i in range(num_cols)]
                               for j in range(num_rows)])

        # returns fuzzy sets characterized by gaussian MF with two parameters, mean and std
        if parametric:
            return np.asarray([[FuzzySet(membership_function_params=[np.mean(np.random.uniform(0, 100, 2)),
                                                                     np.std(np.random.uniform(0, 100, 2))])
                                for i in range(num_cols)]
                               for j in range(num_rows)])

    # TODO profile and compare with
    '''fuzzy_dataset_same = np.full((num_rows, num_cols), 
                              dtype=FuzzySet, 
                              fill_value=FuzzySet(elements=np.random.uniform(0, 100, 10),
                                                  mf=gaussmf,
                                                  params=[np.mean(np.random.uniform(0, 100, 10)),
                                                          np.std(np.random.uniform(0, 100, 10))]))
                                                          '''

    # TODO better parsing

def get_mean_and_std_matrix(fuzzyDataSet: np.ndarray):
    '''
        get the mean (np.matrix) and the std (np.matrix) from fuzzy data (numpy matrix of fuzzy set objects)
        this work only with non-singleton fuzzyfication and gaussian membership parameters
        :return:
     '''
    num_rows, num_cols = fuzzyDataSet.shape
    mean_X = np.asarray([[fuzzyDataSet[j, i].get_membership_function_params()[0] for i in range(num_cols)]  for j in range(num_rows)])
    sigma_X =  np.asarray([[fuzzyDataSet[j, i].get_membership_function_params()[1] for i in range(num_cols)]  for j in range(num_rows)])

    return mean_X,sigma_X


class NonSingletonFuzzifier(BaseEstimator,TransformerMixin):
    def __init__(self, std_proportion=1.0,constant_std=True):
        super(NonSingletonFuzzifier,self).__init__()
        self.std_proportion = std_proportion
        self.constant_std_=constant_std

    def transform(self,X,y=None):
        num_rows = X.shape[0]
        num_cols = X.shape[1]

        return np.asarray([[FuzzySet(membership_function_params=[X[j, i],
                                                                 self.std_proportion])
                            for i in range(num_cols)]
                           for j in range(num_rows)])
        #if self.constant_std_==True:
        #    num_rows = X.shape[0]
        #    num_cols = X.shape[1]

        #   return np.asarray([[FuzzySet(membership_function_params=[X[j, i],
        #                                                            self.std_proportion])
        #                                  for i in range(num_cols)]
        #                                 for j in range(num_rows)])
        #if self.constant_std_ == False:
        #    df = pd.DataFrame(data=X, columns=['x1', 'x2'])
        #    df['y'] = y
        #    fuzzy_data = FuzzyData(data=df, target='y')
        #    fuzzy_data.non_singleton_fuzzification_classification(constant_std=self.constant_std_, std_proportion=self.std_proportion)
        #    return fuzzy_data.get_fuzzydata()

    def fit(self,X, y=None):
        return self
