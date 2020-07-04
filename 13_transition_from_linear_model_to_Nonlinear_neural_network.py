import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=500, n_features = 2, n_informative = 2,
                           n_redundant = 0, n_repeated = 0,
                           n_classes = 2, n_clusters_per_class = 2, class_sep = 0.9,
                           shuffle = True, random_state = 0, flip_y = 0.0)

X, X_test, y, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1234)
plt.scatter(X[:, 0], X[:, 1], s = 60, c = y, cmap = plt.cm.coolwarm)
plt.xlabel("Feature $x-1$", fontsize = 15)
plt.ylabel('Feature $x_2$', fontsize = 15)
plt.title("Data", fontsize = 15)

#plt.show()


def plt_decision_boundaries(X, y, model_class, **model_params):
    reduced_data = X[:, :2]
    model = model_class(**model_params)
    model.fit(reduced_data, y)
    print(classification_report(y, model.predict(X)))

    h = .02

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, y_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z= model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha = 0.4)
    plt.scatter(X[:, 0], X[:, 1], s = 60, c = y, alpha = 0.8, cmap = plt.cm.coolwarm)
    return plt, model

_, model = plt_decision_boundaries(X, y, MLPClassifier)
print("\n\nTest set accuracy\n\n")
print(classification_report(y_test, model.predict(X_test)))

_.show()
