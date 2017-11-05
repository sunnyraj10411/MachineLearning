import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC


names = ["Linear SVM", "Polynomial SVM", "Gaussian SVM"]

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly"),
    SVC(gamma=2, C=1),
	]

# X is the dataset, y is the labels
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
datasets = [(X, y)]
figure = plt.figure(figsize=(27, 9))
i = 1

# Standardize features by removing the mean and scaling to unit variance
X = StandardScaler().fit_transform(X)
# set up the coordinate size
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))

# plot the dataset
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FFF000', '#000FFF'])
ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
           edgecolors='k')

i += 1

# iterate classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    clf.fit(X, y)

    # decision_function is the distance from the points to the decision boundary
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Plot the decision boundary
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the original points
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
               edgecolors='k')
    i += 1

plt.tight_layout()
plt.show()
