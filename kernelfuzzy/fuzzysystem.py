# run block of code and catch warnings


import numpy as np
from kernelfuzzy.fuzzyset import FuzzySet
import matplotlib.pyplot as plt
from  kernelfuzzy.fuzzification import FuzzyData, NonSingletonFuzzifier
from kernelfuzzy.kernels import nonsingleton_gaussian_kernel, gram_matrix_nonsingleton_gaussian_kernel
from kernelfuzzy.kernels import gram_matrix_KBF_kernel,KBFkernel,NonSingletonKernel





def get_rule_antecedents(clf, fuzzy_data, gamma):
    rules = fuzzy_data.copy()
    rules = rules[clf.support_, :]

    # updatate sigmas of MFs of rule antecedents with gamma parameter

    num_rows = rules.shape[0]
    num_cols = rules.shape[1]

    return np.asarray([[FuzzySet(membership_function_params=[rules[j, i].get_membership_function_params()[0],
                                                             rules[j, i].get_membership_function_params()[1] * np.sqrt(1/gamma)])
                        for i in range(num_cols)]
                       for j in range(num_rows)])


def plot_membership_fun(rules_antecedents,X,percentage_range):
    D = rules_antecedents.shape[1]  # number of dimensions

    for d in range(0, D):
        #plt.subplots(figsize=(15, 3))
        plt.subplots()
        params = [fuzzySet.get_membership_function_params() for fuzzySet in rules_antecedents[:, d]]
        range_dim=(np.max(X[:, d]) - np.min(X[:, d]))
        elems = np.arange(np.min(X[:, d])-percentage_range/100*range_dim, np.max(X[:, d])+percentage_range/100*range_dim, range_dim / 1000)
        [plt.plot(elems, np.exp(-0.5 * np.square(elems - mu) / sigma ** 2), 'b', linewidth=1.5) for mu, sigma in params]
        plt.title("dimension : {}".format(d))
        plt.show()


def NSFS_predict(model, X, X_test, option=0):
    # getting best cv parameters
    gamma = model.best_params_['kernel__param']
    std_proportion = model.best_params_['Fuzzifier__std_proportion']
    best_model = model.best_estimator_['svm']

    if option == 0:
        ############################
        # approach 1:  not embedding the gamma value into the nonsingleton fuzzifier
        ############################
        print("First approach")
        fuzzy_data = NonSingletonFuzzifier(std_proportion=std_proportion, constant_std=True).transform(X)
        fuzzy_data_test = NonSingletonFuzzifier(std_proportion=std_proportion, constant_std=True).transform(X_test)
        # KBF/FBF expansion
        K = gram_matrix_KBF_kernel(fuzzy_data_test, fuzzy_data, gamma)
        y_pred = (K[:, best_model.support_] @ best_model.dual_coef_.T + best_model.intercept_).flatten()
        return K, y_pred

    if option == 1:
        ############################
        # approach 2: embedding the gamma value into the nonsingleton fuzzifier
        ############################
        print("Second approach")
        # prediction with a FBF as KBF constructed using the nonsingleton kernel
        # nonsingleton fuzzyfication
        fuzzy_data = NonSingletonFuzzifier(std_proportion=std_proportion * np.sqrt(1 / gamma),
                                           constant_std=True).transform(X)
        fuzzy_data_test = NonSingletonFuzzifier(std_proportion=std_proportion * np.sqrt(1 / gamma),
                                                constant_std=True).transform(X_test)
        # KBF/FBF expansion
        K = gram_matrix_KBF_kernel(fuzzy_data_test, fuzzy_data,
                                   1)  # gamma=1 because we actually embedded the gamma in the fuzzification and in the rules
        y_pred = (K[:, best_model.support_] @ best_model.dual_coef_.T + best_model.intercept_).flatten()
        return K, y_pred

    if option == 2:
        ############################
        # approach 3: embedding the gamma only into the nonsingleton fuzzifier of the input and using the rules
        ############################
        print("Third approach I")
        fuzzy_data = NonSingletonFuzzifier(std_proportion=std_proportion, constant_std=True).transform(X)
        rules_antecedents = get_rule_antecedents(best_model, fuzzy_data, gamma)

        fuzzy_data_test = NonSingletonFuzzifier(std_proportion=std_proportion * np.sqrt(1 / gamma),
                                                constant_std=True).transform(X_test)
        K = gram_matrix_KBF_kernel(fuzzy_data_test, rules_antecedents,1)
        y_pred = (K @ best_model.dual_coef_.T + best_model.intercept_).flatten()
        return K, y_pred

    if option == 3:
        ############################
        # approach 3: embedding the gamma only in the input and using the rules
        ############################
        print("Third approach II")
        rule_antecedents = NonSingletonFuzzifier(std_proportion=std_proportion * np.sqrt(1 / gamma),
                                                 constant_std=True).transform(X[best_model.support_, :])
        fuzzy_data_test = NonSingletonFuzzifier(std_proportion=std_proportion * np.sqrt(1 / gamma),
                                                constant_std=True).transform(X_test)
        K = gram_matrix_KBF_kernel(fuzzy_data_test, rule_antecedents,1)
        y_pred = (K @ best_model.dual_coef_.T + best_model.intercept_).flatten()
        return K, y_pred

    if option == 4:
        ############################
        # prediction of a SVM and the non-singleton kernel
        ############################
        print("SVM + nonsingleton kernel")
        fuzzy_data = NonSingletonFuzzifier(std_proportion=std_proportion, constant_std=True).transform(X)
        fuzzy_data_test = NonSingletonFuzzifier(std_proportion=std_proportion, constant_std=True).transform(X_test)
        # KBF/FBF expansion
        K = gram_matrix_nonsingleton_gaussian_kernel(fuzzy_data_test, fuzzy_data, gamma)
        y_pred = best_model.predict(K)
        return K, y_pred