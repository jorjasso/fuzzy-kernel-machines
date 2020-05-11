"""

    Kernel functions for fuzzy and non-fuzzy sets

"""

import itertools
import numpy as np
import types
from kernelfuzzy.fuzzyset import FuzzySet
from  kernelfuzzy.fuzzification import FuzzyData, get_mean_and_std_matrix
from typing import Callable, List
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy.ma as ma
# from numba.typed import List
from numba import njit,prange
from functools import reduce
from operator import attrgetter


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
    x = [kernel_on_elements(*input_validation(val[0][0], val[1][0], params_kernel_on_elements)) for val in
         cross_product_map]
    y = [kernel_on_membership_degrees(*input_validation(val[0][1], val[1][1], params_kernel_on_membership_degrees)) for
         val in cross_product_map]

    x = np.asarray([float(i) for i in x])
    y = np.asarray([float(i) for i in y])
    return np.dot(x, y)


def input_validation(x: np.ndarray, y: np.ndarray, params: List[float] = ''):
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
    if sum(np.array(x).shape) == 0 or sum(np.array(x).shape) == 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    # unique observation with multiple features:  np.array([3,3]) for example
    if x.shape[0] > 1 & len(x.shape) == 1:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

    # multiple observations with unique features
    if x.shape[0] > 1 & len(x.shape) > 1:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    arguments = [x, y]

    if type(params) is list:
        for e in params:
            arguments.append(e)
    if params != '' and type(params) is not list:
        arguments.append(params)

    return tuple(arguments)


def gram_matrix_cross_product_kernel(X: FuzzyData,
                                     Y: FuzzyData,
                                     kernel_on_elements: Callable,
                                     params_kernel_on_elements: List,
                                     kernel_on_membership_degrees: Callable,
                                     params_kernel_on_membership_degrees: List) -> np.ndarray:
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


# @njit

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


@njit(parallel=True, nogil=True, cache=True)
def gram_matrix_nonsingleton_gaussian_kernel_njit(mean_X: np.ndarray,
                                                  mean_Y: np.ndarray,
                                                  sigma_X: np.ndarray,
                                                  sigma_Y: np.ndarray,
                                                  param: float) -> np.ndarray:
    '''

        Calculates the Gram matrix using the nonsingleton Gaussian kernel on fuzzy sets
        optimized for Numba

        Input:
            mean_X, sigma_X:     matrix of parameters for fuzzy data X (Type: FuzzyData)
            mean_Y, sigma_Y:     matrix of parameters for fuzzy data Y (Type: FuzzyData)
            param:                  (Type: float) kernel parameter

        Output:
            (Type: numpy.ndarray) kernel matrix

        '''

    gram_matrix = np.zeros((mean_X.shape[0], mean_Y.shape[0]))


    if mean_X is mean_Y:  # use symmetry property
        N = mean_X.shape[0]

        #for i in range(0, N):
        for i in prange(N):
            for j in range(i, N):

                value = 1
                for mean_x, mean_y, sigma_x, sigma_y in zip(mean_X[i, :], mean_Y[j, :], sigma_X[i, :], sigma_Y[j, :]):
                    value = value * (np.exp(-0.5 * param * (mean_x - mean_y) ** 2 / (sigma_x ** 2 + sigma_y ** 2)))

                gram_matrix[i, j] = value
                gram_matrix[j, i] = value
        # tikonov regularization
        #gram_matrix = gram_matrix + np.eye(N) * np.nextafter(0, 1)


    else:  # X and Y are different
        N = mean_X.shape[0]
        M = mean_Y.shape[0]

        #for i in range(0, N):
        for i in prange(N):
            for j in range(0, M):

                value = 1
                for mean_x, mean_y, sigma_x, sigma_y in zip(mean_X[i, :], mean_Y[j, :], sigma_X[i, :], sigma_Y[j, :]):
                    value = value * (np.exp(-0.5 * param * (mean_x - mean_y) ** 2 / (sigma_x ** 2 + sigma_y ** 2)))

                gram_matrix[i, j] = value

    return gram_matrix


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

    #
    mean_X, sigma_X = get_mean_and_std_matrix(X)

    if X is Y:  # use symmetry property
        N = mean_X.shape[0]
        gram_matrix = gram_matrix_nonsingleton_gaussian_kernel_njit(mean_X, mean_X, sigma_X, sigma_X, 1) + np.eye(N) * np.nextafter(0, 1)
    else:  # X and Y are different
        mean_Y, sigma_Y = get_mean_and_std_matrix(Y)
        gram_matrix = gram_matrix_nonsingleton_gaussian_kernel_njit(mean_X, mean_Y, sigma_X, sigma_Y, 1)

    '''
        gram_matrix = np.zeros((X.shape[0], Y.shape[0]))

        if X is Y: # use symmetry property
            N=X.shape[0]

            for i in range(0,N):
                for j in range(i, N):

                    value = 1
                    for x, y, in zip(X[i,:], X[j,:]):
                        value = value * nonsingleton_gaussian_kernel(x, y, param)

                    gram_matrix[i, j] = value
                    gram_matrix[j, i] = value
            #tikonov regularization
            gram_matrix=gram_matrix+np.eye(N)*np.nextafter(0,1)


        else: # X and Y are different

            for i, tuple_x in enumerate(X):
                for j, tuple_y in enumerate(Y):

                    #
                    value=1
                    for x, y, in zip(tuple_x, tuple_y):
                        value=value*nonsingleton_gaussian_kernel(x, y, param)

                    gram_matrix[i, j] = value

        
        '''
    return gram_matrix



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

    gram_matrix = gram_matrix_nonsingleton_gaussian_kernel(X, Y, KBF_param)

    # because overflow
    denominator = 1 / (np.sum(gram_matrix, axis=1) + np.nextafter(0, 1))
    denominator = np.nan_to_num(denominator)

    gram_matrix = gram_matrix * denominator[:, np.newaxis]

    return gram_matrix


# Wrapper class for the custom kernel KBF kernel
class KBFkernel(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        super(KBFkernel, self).__init__()
        self.param = param

    def transform(self, X):
        return gram_matrix_KBF_kernel(X, self.X_train_, KBF_param=self.param)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


class KBFkernelSymmetric(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        super(KBFkernelSymmetric, self).__init__()
        self.param = param

    def transform(self, X):
        return gram_matrix_KBF_kernel(X, self.X_train_, KBF_param=self.param) + np.transpose(
            gram_matrix_KBF_kernel(self.X_train_, X, KBF_param=self.param)) / 2

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self


# Wrapper class for the custom kernel KBF kernel
class NonSingletonKernel(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        super(NonSingletonKernel, self).__init__()
        self.param = param

    def transform(self, X):
        return gram_matrix_nonsingleton_gaussian_kernel(X, self.X_train_, param=self.param)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self
