import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import mixture

color_iter = itertools.cycle(["black", "red", "purple"])

# sci-kit learn
def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(1, 7)
    plt.ylim(-0., 3.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


dataPd = pd.read_csv('iris.data',delimiter=',',names=['0','1'])
data = np.asarray(dataPd)

# Fit a Gaussian mixture with EM 
gmm = mixture.GaussianMixture(n_components=3).fit(data)
plot_results(data, gmm.predict(data), gmm.means_, gmm.covariances_, 0,
             'Gaussian Mixture')


plt.show()
