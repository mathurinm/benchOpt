import numpy as np
from scipy import sparse

from benchopt.base import BaseSolver
from benchopt.util import safe_import_context


with safe_import_context() as import_ctx:

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    from benchopt.utils.r_helpers import import_rpackages

    # Setup the system to allow rpy2 running
    numpy2ri.activate()
    import_rpackages('glmnet')


class Solver(BaseSolver):
    name = "glmnet"

    install_cmd = 'conda'
    requirements = ['r-base', 'rpy2', 'r-glmnet', 'r-matrix']
    stop_strategy = 'iteration'
    support_sparse = True

    def set_objective(self, X, y, lmbd):
        self.y, self.lmbd = y, lmbd
        self.lmbd_max = np.max(np.abs(X.T @ y))
        self.glmnet = robjects.r['glmnet']

        if sparse.issparse(X):
            r_Matrix = packages.importr("Matrix")
            self.X = r_Matrix.sparseMatrix(
                i = robjects.IntVector(X.row + 1),
                j = robjects.IntVector(X.col + 1),
                x = robjects.FloatVector(X.data),
                dims = robjects.IntVector(X.shape)
            )
        else:
            self.X = X

    def run(self, n_iter):
        fit_dict = {"lambda.min.ratio": self.lmbd / self.lmbd_max}
        glmnet_fit = self.glmnet(self.X, self.y, intercept=False,
                                 standardize=False, maxit=n_iter,
                                 thresh=1e-14, **fit_dict)
        results = dict(zip(glmnet_fit.names, list(glmnet_fit)))
        as_matrix = robjects.r['as']
        coefs = np.array(as_matrix(results["beta"], "matrix"))
        self.w = coefs[:, -1]

    def get_result(self):
        return self.w
