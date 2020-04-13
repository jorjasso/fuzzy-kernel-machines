import numpy as np
from kernelfuzzy.fuzzyset import FuzzySet
import matplotlib.pyplot as plt



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