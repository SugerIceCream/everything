from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def Linear():
    x, y = make_regression(n_samples=50, n_features=1, n_informative=1, noise=50, random_state=1)
    reg = LinearRegression()
    reg.fit(x, y)
    z = np.linspace(-3, 3, 200).reshape(-1,1)
    plt.scatter(x, y, c='b', s=60)
    plt.plot(z, reg.predict(z), c='k')
    plt.title('Linear Regression')
    plt.show()

Linear()