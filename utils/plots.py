import plotly.offline as pltoff
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kernelfuzzy.fuzzification import FuzzyData
from kernelfuzzy.kernels import gram_matrix_KBF_kernel


def plot1D(fuzzyset):

    """

    Draws a 2D plot of a 1D fuzzy set and its membership degrees

    Input:
        fuzzyset: (Type: object "FuzzySet") a fuzzy set

    Output:
        None

    """

    trace = go.Scatter(
        text = "Degrees",
        x = fuzzyset.get_set(),
        y = fuzzyset.get_membership_degrees(),
        mode = 'markers'
    )

    data = [trace]

    layout = dict(
        title = 'Fuzzy sets and its membership degrees',
        xaxis = dict(title = 'Elements'),
        yaxis = dict(title = 'Degrees'),
    )

    fig = dict(data=data, layout=layout)
    
    pltoff.iplot(fig, filename='fuzzyset.html')


def plot2D(fuzzyset):

    """

    Draws a 3D plot of a 2D set and its membership degrees

    Input:
        set:     (Type: numpy.array)   a 2D fuzzy set
        degrees: (Type: list of reals) membership degrees of the set

    Output:
        None

    """
    pass
    
def plot3D(fuzzyset):

    """

    Draws a 4D plot of a 3D set and its membership degrees

    Input:
        set:     (Type: numpy.array)   a 3D fuzzy set
        degrees: (Type: list of reals) membership degrees of the set

    Output:
        None

    """
    pass

####
def plot_decision_function_kernel(X, X_fuzzy, y, clf):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    df = pd.DataFrame(data=xy, columns=['x1', 'x2'])
    df['y'] = 1
    fuzzy_test_data = FuzzyData(data=df, target='y')
    fuzzy_test_data.non_singleton_fuzzification_classification(constant_std=False)

    x_test = fuzzy_test_data.get_fuzzydata()

    # training
    K = gram_matrix_KBF_kernel(x_test, X_fuzzy, 1)

    Z = clf.decision_function(K).reshape(XX.shape)

    # plot support vectors
    ax.scatter(X[clf.support_, 0], X[clf.support_, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    # plot decision boundary and margins
    CS = ax.contour(XX, YY, Z, colors='k',  alpha=0.5)
    ax.clabel(CS, inline=1, fontsize=10)

    plt.show()


def plot_decision_function(X, y, clf):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot support vectors
    ax.scatter(X[clf.support_, 0], X[clf.support_, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    plt.show()
    