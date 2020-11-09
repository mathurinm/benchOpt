import numpy as np

from benchopt.base import BaseDataset
from benchopt.utils.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (100, 5000),
            (100, 10000)
        ], 'rho': [0, 0.6]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=27, rho=0):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.rho = rho

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        if self.rho == 0:
            X = rng.randn(self.n_samples, self.n_features)
        else:
            X = make_correlated_data(self.n_samples, self.n_features,
                                     rho=self.rho, random_state=rng)

        y = rng.randn(self.n_samples)

        data = dict(X=X, y=y)

        return self.n_features, data
