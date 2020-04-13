"""

    Kernel functions for fuzzy and non-fuzzy sets

"""

import itertools
import numpy as np
import types
from kernelfuzzy.fuzzyset import FuzzySet
from kernelfuzzy.fuzzification import FuzzyData
from typing import Callable, List
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy.ma as ma

def linear_kernel(X, Y):
    
    """

    the linear kernel
    
    Input:
        X: (Type: FuzzySet)
        Y: (Type: FuzzySet) 

    Output:
        (Type: real) kernel result

    """
    
    X = np.array(X)
    Y = np.array(Y)

    return np.dot(X, Y.T)

def cross_product_kernel(X: FuzzySet,
                         Y: FuzzySet,
                         kernel_on_elements: Callable,
                         params_kernel_on_elements: List,
                         kernel_on_membership_degrees: Callable,
                         params_kernel_on_membership_degrees: List) -> np.ndarray:
    """

    Calculates the Cross Product kernel between two fuzzy sets X and Y

    Input:
        X:                      (Type: FuzzySet)
        Y:                      (Type: FuzzySet)
        kernel_elements:        (Type: Callable)
        params_kernel_elements: (Type: List)
        kernel_degrees:         (Type: Callable)
        params_kernel_degrees:  (Type: List)

    Output:
        (Type: numpy.ndarray) kernel result

    """

    # create cross-product map
    x = X.get_pair()
    y = Y.get_pair()

    cross_product_map = list(itertools.product(*list([x, y])))
    
    # iterate over the cross-product map
    x = [kernel_on_elements(*input_validation(val[0][0], val[1][0], params_kernel_on_elements)) for val in cross_product_map]
    y = [kernel_on_membership_degrees(*input_validation(val[0][1], val[1][1], params_kernel_on_membership_degrees)) for val in cross_product_map]

    x = np.asarray([float(i) for i in x])
    y = np.asarray([float(i) for i in y])
    return np.dot(x, y)


def input_validation(x: np.ndarray, y: np.ndarray, params: List[float]=''):
    
    """

    Argument validation to be used by sklearn methods.
    By convention rows are observations and columns features.

    Input:
        x:      (Type: np.ndarray)
        y:      (Type: np.ndarray)
        params: (Type: list)

    Output:
        (Type: tuple)

    """
    
    x = np.array(x)
    y = np.array(y)
    
    # unique observation with unique feature
    #     sum(np.array(3).shape) prints 0; sum(np.array([3]).shape) prints 1
    if sum(np.array(x).shape) == 0 or sum(np.array(x).shape) == 1 :
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    # unique observation with multiple features:  np.array([3,3]) for example
    if x.shape[0] > 1 & len(x.shape)==1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

    # multiple observations with unique features
    if x.shape[0]>1 & len(x.shape)>1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    arguments=[x,y]

    if type(params) is list:
        for e in params:
            arguments.append(e)
    if params != '' and type(params) is not list:
        arguments.append(params)

    return tuple(arguments)


def gram_matrix_cross_product_kernel(X: FuzzyData,
                                     Y:FuzzyData,
                                     kernel_on_elements: Callable,
                                     params_kernel_on_elements: List,
                                     kernel_on_membership_degrees: Callable,
                                     params_kernel_on_membership_degrees: List)-> np.ndarray:
    
    '''

    Calculates the Gram matrix using the Cross Product kernel on  fuzzy sets

    Input:
        X:                      (Type: FuzzyData)
        Y:                      (Type: FuzzyData)
        kernel_elements:        (Type: Callable)
        params_kernel_elements: (Type: List)
        kernel_degrees:         (Type: Callable)
        params_kernel_degrees:  (Type: List)

    Output:
        (Type: numpy.ndarray) kernel matrix

    '''

    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, tuple_x in enumerate(X):
        for j, tuple_y in enumerate(Y):

            # dot-product like operation between tuple of fuzzy sets
            value = 0
            for x, y, in zip(tuple_x, tuple_y):
                value = value + cross_product_kernel(x, y,
                                                     kernel_on_elements,
                                                     params_kernel_on_elements,
                                                     kernel_on_membership_degrees,
                                                     params_kernel_on_membership_degrees)
            gram_matrix[i, j] = value
    return gram_matrix


def nonsingleton_gaussian_kernel(X: FuzzySet,
                                 Y: FuzzySet,
                                 gamma: float) -> np.ndarray:
    """

    Calculates the non-singleton Gaussian kernel between two fuzzy sets X and Y

    Input:
        X:                      (Type: FuzzySet)
        Y:                      (Type: FuzzySet)
        gamma:                  (Type: float) kernel parameter

    Output:
        (Type: numpy.ndarray) kernel result

    """

    # get Gaussian parameters
    mean_x, sigma_x = X.get_membership_function_params()
    mean_y, sigma_y = Y.get_membership_function_params()

    return np.exp(-0.5 * gamma * (mean_x - mean_y) ** 2 / (sigma_x ** 2 + sigma_y ** 2))

def gram_matrix_nonsingleton_gaussian_kernel(X: FuzzyData,
                                             Y: FuzzyData,
                                             param: float) -> np.ndarray:

        '''

        Calculates the Gram matrix using the nonsingleton Gaussian kernel on fuzzy sets

        Input:
            X:                      (Type: FuzzyData)
            Y:                      (Type: FuzzyData)
            gamma:                  (Type: float) kernel parameter

        Output:
            (Type: numpy.ndarray) kernel matrix

        '''

        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i, tuple_x in enumerate(X):
            for j, tuple_y in enumerate(Y):

                #
                value=1
                for x, y, in zip(tuple_x, tuple_y):
                    value=value*nonsingleton_gaussian_kernel(x, y, param)

                #TODO profile this
                #value=0
                #for x, y, in zip(tuple_x, tuple_y):
                #    value=value+test(x,y,gamma)
                #value=np.exp(value)
                #print('(prod, value)', prod, value)

                gram_matrix[i, j] = value
        if X is Y:
            gram_matrix=gram_matrix+np.eye(X.shape[0])*np.nextafter(0,1)
        return gram_matrix

#TODO profile this
#def test(X: FuzzySet,Y: FuzzySet, gamma: float):
#    mean_x, sigma_x = X.get_membership_function_params()
#    mean_y, sigma_y = Y.get_membership_function_params()

#    return -0.5 * gamma * (mean_x - mean_y) ** 2 / (sigma_x ** 2 + sigma_y ** 2)

def gram_matrix_KBF_kernel(X: FuzzyData,
                           Y: FuzzyData,
                           KBF_param: float) -> np.ndarray:
    '''

    Calculates the Gram matrix using the nonsingleton Gaussian kernel on fuzzy sets

    Input:
        X:                      (Type: FuzzyData)
        Y:                      (Type: FuzzyData)
        gamma:                  (Type: float) kernel parameter

    Output:
        (Type: numpy.ndarray) kernel matrix

    '''

    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))
    for i, tuple_x in enumerate(X):
        for j, tuple_y in enumerate(Y):

            #
            value = 1
            for x, y, in zip(tuple_x, tuple_y):
                value = value * nonsingleton_gaussian_kernel(x, y, KBF_param)

            gram_matrix[i, j] = value

    if X is Y:
        gram_matrix = gram_matrix + np.eye(X.shape[0]) * np.nextafter(0, 1)

    #because overflow
    denominator=1/(np.sum(gram_matrix, axis=1)+np.nextafter(0, 1))
    denominator=np.nan_to_num(denominator)


    #retrieve kernel parameters and data associated with invalid values and implement an strategy for those values
    #print("gram_matrix :{}".format(gram_matrix))
    #print("np.sum(gram_matrix, axis=1) :{}".format(np.sum(gram_matrix, axis=1)))
    #print("denominator",denominator)

    #print("details:")
    #
    #for i, tuple_x in enumerate(X):
    #    for j, tuple_y in enumerate(Y):

    #        if i==0 and j==0:
    #            value = 1
    #            for x, y, in zip(tuple_x, tuple_y):
    #                print("fuzzy sets x, y")
    #                print("x : {}".format(x.get_membership_function_params()))
    #                print("y : {}".format(y.get_membership_function_params()))
    #                print("nonsingleton_gaussian_kernel(x, y, KBF_param) : {}".format(nonsingleton_gaussian_kernel(x, y, KBF_param)))
    #                value = value * nonsingleton_gaussian_kernel(x, y, KBF_param)
    #                print("accumulative value : {}".format(value))




    #
    #masked_K = np.ma.masked_invalid(gram_matrix)
    #print("invalid values",gram_matrix[masked_K.mask])
    #print("kernel parameter : {}".format(KBF_param))
    #idx=np.where(masked_K.mask==True)
    #print("invalid values", gram_matrix[idx])
    #print("associated values X: {}".format(X[idx[0]]))
    #print("associated values Y: {}".format(X[idx[1]]))

    #masked_denominator=np.ma.masked_invalid(denominator)
    #idx=np.where(masked_denominator.mask==True)
    #print("invalid values denominator : {}".format(denominator[idx]))


    #         print(K)
    #         print(masked_K)
    #         print(masked_K.mask)
    #         print(np.where(masked_K.mask==True))
    #         print(K[masked_K.mask])
    #

    gram_matrix=gram_matrix*denominator[:, np.newaxis]

    #if gram_matrix.shape[0]== gram_matrix.shape[1]:
    #    gram_matrix=(gram_matrix+np.transpose(gram_matrix))/2


    return gram_matrix

# Wrapper class for the custom kernel KBF kernel
class KBFkernel(BaseEstimator,TransformerMixin):
    def __init__(self, param=1.0):
        super(KBFkernel,self).__init__()
        self.param = param

    def transform(self, X):
        return gram_matrix_KBF_kernel(X, self.X_train_, KBF_param=self.param)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


# Wrapper class for the custom kernel KBF kernel
class NonSingletonKernel(BaseEstimator,TransformerMixin):
    def __init__(self, param=1.0):
        super(NonSingletonKernel,self).__init__()
        self.param = param

    def transform(self, X):
        return gram_matrix_nonsingleton_gaussian_kernel(X, self.X_train_, param=self.param)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self
