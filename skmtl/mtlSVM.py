"""
This is a module to develop the classes of Multi-Task Learning SVMs
"""
from sklearn.svm import SVC
from sklearn.svm import SVR
import sklearn.metrics.pairwise as pairwise
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, r2_score
import numpy_indexed as npi
import types


def neg_r2_score(y_true, y_pred, sample_weight=None):
    return -r2_score(y_true, y_pred, sample_weight)


class MTLSVM(BaseEstimator):
    """ A Multi-Task SVM base estimator.

    The MTLSVM can be fitted as a standard SVM using a specific choice
    of kernel function. This class performs the operations needed to
    compute the MTL kernel and the training is then left for standard
    SVMs.     
    We consider two MTL approaches:
    - Joint Learning of common and task specific models: here we
        consider that for the final prediction we combine a common and
        task-specific predictions. The corresponding models can be
        defined in different Hilbert spaces.
    - Graph Laplacian Regularization: here all the task parameters
        belong in the same Hilbert space . The goal is to impose
        similarity among task models by penalizing the weighted sum of
        distances between the task parameters and the weights are
        given by an adjacency matrix A of a task-based graph.

        See "Learning Multiple Tasks with Kernel Methods". Evgeniou et al.
        and ""

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.

    Examples
    --------
    >>> from skmtl import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """
    def __init__(self, C=1.0, ckernel='rbf', skernel='rbf', degree=3,
                 cgamma='auto', sgamma='auto', coef0=0.0, shrinking=True,
                 tol=0.001, cache_size=200, verbose=False,
                 max_iter=-1, mu=1.0, mu2=1.0, task_info=None, mtl_type='convex',  delta='auto', order_delta='auto'):
        super(MTLSVM, self).__init__()
        self.C = C
        self.ckernel = ckernel
        self.skernel = skernel
        self.degree = degree
        self.cgamma = cgamma
        self.sgamma = sgamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.mu = mu
        self.mu2 = mu2
        self.task_info = task_info
        self.mtl_type = mtl_type
        self.deltainv = None
        if isinstance(mu, dict) or isinstance(mu, list):
            self.sparam=True # Pasar como parámetro?
        else:
            self.sparam = False
        self.delta = delta
        self.order_delta = order_delta

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        self.X_train = X
        n, m = X.shape

        if self.task_info is None:
            self.task_info = task_info
        task_col = self.task_info
        self.unique, self.groups_idx = npi.group_by(X[:, task_col],
                                                    np.arange(n))
        if 'fused' in self.mtl_type or self.mtl_type == 'graphLap':
            if 'delta' in kwargs:
                self.delta = kwargs['delta']
            else:
                T = len(self.unique)
                delta = np.ones((T, T))
                for i in range(T):
                    delta[i, i] = 0
                    delta[i, :] /= -np.sum(delta[i, :])
                    delta[i, i] = -np.sum(delta[i, :])
                self.delta = delta / np.sum(np.abs(delta))
        if self.mtl_type == 'fused' and 'order_delta' in kwargs:
            self.order_delta = kwargs['order_delta']
        if self.cgamma == 'auto':
            self.cgamma = 1. / (m-1)
        else:
            self.cgamma = self.cgamma
        if self.sgamma == 'auto':
            self.sgamma = self.cgamma
        else:
            self.sgamma = self.sgamma
        
        G_train = self._mtl_kernel(X, X, self.ckernel, self.skernel,
                                   task_info, self.mu)
        # if y.ndim == 1:
        #     y_2d = y.reshape(-1, 1)
        # else:
        #     y_2d = y
        self.y = y
        self.svm.fit(G_train, self.y, sample_weight)
        # print(G_train)
        # print(self.svm)
        self.support_ = self.svm.support_
        self.support_vectors_ = self.svm.support_vectors_
        self.dual_coef_ = self.svm.dual_coef_
        # self.coef_ = self.svm.coef_
        self.intercept_ = self.svm.intercept_
        self.sample_weight = sample_weight
        self.task_info = task_info

    def predict(self, X):
        G_test = self._mtl_kernel(X, self.X_train, self.ckernel, self.skernel,
                                  self.task_info, self.mu, cgamma=self.cgamma,
                                  sgamma=self.sgamma)
        # print(G_test)
        return self.svm.predict(G_test)

    def score(self, X, y, sample_weight=None, scoring=None):
        G_test = self._mtl_kernel(X, self.X_train, self.ckernel, self.skernel,
                                  self.task_info, self.mu, cgamma=self.cgamma,
                                  sgamma=self.sgamma)
        y_pred = self.svm.predict(G_test)

        n, m = X.shape
        task_col = self.task_info
        unique, groups_idx = npi.group_by(X[:, task_col],
                                          np.arange(n))
        self.scores = {}
        for i, t in enumerate(unique):
            y_true_g = y[groups_idx[i]]
            y_pred_g = y_pred[groups_idx[i]]
            if scoring is None:
                self.scores[t] = self.score_fun(y_true_g, y_pred_g)
            else:
                self.scores[t] = scoring(y_true_g, y_pred_g)

        if scoring is None:
            return self.score_fun(y, y_pred, sample_weight)
        else:
            return scoring(y, y_pred, sample_weight)

    def _get_kernel_fun(self, kernel):
        # if not isinstance(kernel, (str, types.FunctionType)):
        #     raise Exception('kernel of wrong type')
        if isinstance(kernel, str):
            kernel_f = getattr(pairwise, kernel+'_kernel')
        else:
            kernel_f = kernel
        return kernel_f

    def _apply_kernel(self, kernel, x, y, **kwargs):
        kernel_f = self._get_kernel_fun(kernel)
        if kernel_f == pairwise.rbf_kernel:
            if 'gamma' not in kwargs:
                if kwargs['common']:
                    gamma = kwargs['cgamma']
                else:
                    if 'task' not in kwargs:
                        gamma = kwargs['sgamma']
                    else:
                        if kwargs['task'] in kwargs['sgamma']:
                            gamma = kwargs['sgamma'][kwargs['task']]
                        else:
                            gamma = kwargs['sgamma'][float(kwargs['task'])]
            else:
                gamma = kwargs['gamma']
            return kernel_f(x, y, gamma)
        else:
            return kernel_f(x, y)

    def _compute_K(self, x, y, tx, ty, skernel, **kwargs):
        skernel_isList = isinstance(skernel, (list, np.ndarray))
        skernel_isDic = isinstance(skernel, dict)
        if tx == ty:
            if skernel_isList:
                itx = np.where(self.unique == tx)[0][0]
                ret = self._apply_kernel(skernel[itx], x, y,
                                         **dict(kwargs, common=False,
                                                task=itx))
            elif skernel_isDic:
                ret = self._apply_kernel(skernel[tx], x, y,
                                         **dict(kwargs, common=False))
                # if type(tx) is not str:
                #     ret = self._apply_kernel(skernel['%d' % tx], x, y,
                #                              **dict(kwargs, common=False))
                # else:
                #     ret = self._apply_kernel(skernel[tx], x, y,
                #                              **dict(kwargs, common=False))
            else:
                sgamma_isDic = isinstance(kwargs['sgamma'], dict)
                if sgamma_isDic:
                    ret = self._apply_kernel(skernel, x, y,
                                            **dict(kwargs, common=False, task=tx))
                else:
                    ret = self._apply_kernel(skernel, x, y,
                                    **dict(kwargs, common=False))
        else:
            ret = 0
        return ret

    def _compute_Q(self, x, y, tx, ty, ckernel, **kwargs):
        return self._apply_kernel(ckernel, x, y, **dict(kwargs, common=True))

    def _mtl_kernel(self, X, Y, ckernel, skernel, task_info=0, mu=1, **kwargs):
        if self.mtl_type == 'additive':
            return self._mtl_kernel_additive(X, Y, ckernel, skernel, task_info, kwargs, cgamma=self.cgamma,
                                   sgamma=self.sgamma)
        if self.mtl_type == 'convex' or self.mtl_type == 'convex-optlamb':
            self.lamb = self.mu
            return self._mtl_kernel_convex(X, Y, ckernel, skernel, task_info,
                                                    cgamma=self.cgamma, sgamma=self.sgamma)
        elif self.mtl_type == 'fused':
            return self._mtl_kernel_fused(X, Y, ckernel, skernel, task_info, cgamma=self.cgamma,
                                   sgamma=self.sgamma)
        elif self.mtl_type == 'graphLap':
            return self._mtl_kernel_graphLap(X, Y, ckernel, skernel, task_info, cgamma=self.cgamma,
                                   sgamma=self.sgamma)
        elif self.mtl_type == 'altfused':
            return self._mtl_kernel_altfused(X, Y, ckernel, skernel, task_info, cgamma=self.cgamma,
                                   sgamma=self.sgamma)
        elif self.mtl_type == 'fused+common':
            return self._mtl_kernel_fusedAndCommon(X, Y, ckernel, skernel, task_info,
                                                   cgamma=self.cgamma, sgamma=self.sgamma)
        elif self.mtl_type == 'fused+specific':
            return self._mtl_kernel_fusedAndSpecific(X, Y, ckernel, skernel, task_info,
                                                     cgamma=self.cgamma, sgamma=self.sgamma)
        else:
            # print('{} is not a valid mtl type'.format(self.mtl_type))
            exit()


    
    def _mtl_kernel_additive(self, X, Y, ckernel, skernel, task_info=0, mu=1, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        K = np.zeros((nX, nY))

        for i, tx in enumerate(unique_X):
            arr_j = np.where(unique_Y == tx)[0]
            if len(arr_j) == 1:
                j = arr_j[0]
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                K[indX[:, None], indY] = self._compute_K(X_data[indX],
                                                         Y_data[indY],
                                                         tx, tx,
                                                         skernel, **kwargs)
        # for i in range(nX):
        #     for j in range(nY):
        #         K[i, j] = self._compute_K(X_data[i].reshape(1, -1),
        #                                   Y_data[j].reshape(1, -1),
        #                                   task_X[i], task_Y[j],
        #                                   skernel, **kwargs)
        Q = self._apply_kernel(ckernel, X_data, Y_data, **dict(kwargs,
                               common=True))
        
        hat_Q = Q/self.mu + K
        return hat_Q
    
    def _mtl_kernel_convex(self, X, Y, ckernel, skernel, task_info, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        K = np.zeros((nX, nY))
        Q = np.zeros((nX, nY))

        for i, tx in enumerate(unique_X):
            if self.sparam:
                if isinstance(tx, float):
                    tx = int(tx)
                lambX = self.lamb[str(tx)]
            else:
                lambX = self.lamb
            for j, ty in enumerate(unique_Y):
                if self.sparam:
                    if isinstance(ty, float):
                        ty = int(ty)
                    lambY = self.lamb[str(ty)]
                else:
                    lambY = self.lamb
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                Q[indX[:, None], indY] = lambX * lambY * self._compute_Q(X_data[indX],
                                                                         Y_data[indY],
                                                                         tx, ty,
                                                                         ckernel, **kwargs)
                if tx == ty:
                    K[indX[:, None], indY] = (1-lambX)*(1-lambY)*self._compute_K(X_data[indX],
                                                                                 Y_data[indY],
                                                                                 tx, ty,
                                                                                 skernel, **kwargs)
        hat_Q = Q + K
        return hat_Q

    def _mtl_kernel_convex2(self, X, Y, ckernel, skernel, task_info, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        K = np.zeros((nX, nY))
        Q = np.zeros((nX, nY))

        for i, tx in enumerate(unique_X):
            if self.sparam:
                lambX = self.lamb[tx]
                s = np.sum([self.lamb[u]*g for u, g in zip(unique_X, groups_idx_X)])/nX
            else:
                lambX = self.lamb
                s = lambX
            for j, ty in enumerate(unique_Y):
                if self.sparam:
                    lambY = self.lamb[ty]
                else:
                    lambY = self.lamb
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                if s != 0:
                    common_factor =  lambX * lambY / s
                else:
                    common_factor = 0
                Q[indX[:, None], indY] = common_factor * self._compute_Q(X_data[indX],
                                                                         Y_data[indY],
                                                                         tx, ty,
                                                                         ckernel, **kwargs)
                if tx == ty:
                    # sp_factor =  (nX/len(np.argwhere(indX)))*(1-lambX)*(1-lambY)
                    sp_factor =  (nX/len(np.argwhere(indX)))*(1-lambX)
                    K[indX[:, None], indY] = sp_factor * self._compute_K(X_data[indX],
                                                                         Y_data[indY],
                                                                         tx, ty,
                                                                         skernel, **kwargs)
        hat_Q = Q + K
        return hat_Q

    def _mtl_kernel_fused(self, X, Y, ckernel, skernel, task_info=-1, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info
        mu = self.mu

        delta_inc = self.delta
        delta = mu * delta_inc + np.identity(delta_inc.shape[0])
        order_delta = self.order_delta
        if self.deltainv is None:
            self.deltainv = np.linalg.inv(delta)
        

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                if tx in order_delta:
                    order_tx = order_delta[tx]
                else:
                    order_tx = order_delta[float(tx)]
                if ty in order_delta:
                    order_ty = order_delta[ty]
                else:
                    order_ty = order_delta[float(ty)]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy

        Q = self._apply_kernel(ckernel, X_data, Y_data, **dict(kwargs,
                               common=True))
        return np.multiply(A, Q)

    def _mtl_kernel_altfused(self, X, Y, ckernel, skernel, task_info=-1, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info
        mu = self.mu

        delta_inc = self.delta
        T = self.delta.shape[0]
        delta = mu * delta_inc + (1/T**2) * np.ones(delta_inc.shape)
        order_delta = self.order_delta
        if self.deltainv is None:
            self.deltainv = np.linalg.inv(delta)
        

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                if tx in order_delta:
                    order_tx = order_delta[tx]
                else:
                    order_tx = order_delta[float(tx)]
                if ty in order_delta:
                    order_ty = order_delta[ty]
                else:
                    order_ty = order_delta[float(ty)]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy

        Q = self._apply_kernel(ckernel, X_data, Y_data, **dict(kwargs,
                               common=True))
        return np.multiply(A, Q)


    def _mtl_kernel_graphLap(self, X, Y, ckernel, skernel, task_info=-1, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info
        mu = self.mu

        delta_inc = self.delta
        delta = mu * delta_inc
        order_delta = self.order_delta
        if self.deltainv is None:
            self.deltainv = np.linalg.pinv(delta)
        

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                if tx in order_delta:
                    order_tx = order_delta[tx]
                else:
                    order_tx = order_delta[float(tx)]
                if ty in order_delta:
                    order_ty = order_delta[ty]
                else:
                    order_ty = order_delta[float(ty)]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy

        Q = self._apply_kernel(ckernel, X_data, Y_data, **dict(kwargs,
                               common=True))
        return np.multiply(A, Q)

    def _mtl_kernel_fusedAndCommon(self, X, Y, ckernel, skernel, task_info, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info
        mu = self.mu2

        delta_inc = self.delta
        delta = mu * delta_inc + np.identity(delta_inc.shape[0])
        order_delta = self.order_delta
        if self.deltainv is None:
            self.deltainv = np.linalg.inv(delta)
        

        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                if tx in order_delta:
                    order_tx = order_delta[tx]
                else:
                    order_tx = order_delta[float(tx)]
                if ty in order_delta:
                    order_ty = order_delta[ty]
                else:
                    order_ty = order_delta[float(ty)]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy

        Q = self._apply_kernel(ckernel, X_data, Y_data, **dict(kwargs,
                               common=True))

        K = self._apply_kernel(skernel, X_data, Y_data, **dict(kwargs,
                               gamma=self.cgamma))

        hat_Q = self.mu**2 * Q + (1-self.mu)**2 * np.multiply(A, K)
        return hat_Q

    def _mtl_kernel_fusedAndSpecific(self, X, Y, ckernel, skernel, task_info, **kwargs):
        """
        We create a custom kernel for multitask learning.
            If task_info is a scalar it is assumed to be the column of the task
            If task_info is an array it is assumed to be the task indexes
        """
        task_col = task_info
        mu = self.mu2

        delta_inc = self.delta
        delta = mu * delta_inc + np.identity(delta_inc.shape[0])
        order_delta = self.order_delta
        if self.deltainv is None:
            self.deltainv = np.linalg.inv(delta)
        
        X_data = np.delete(X, task_col, axis=1).astype(float)
        Y_data = np.delete(Y, task_col, axis=1).astype(float)
        nX = X_data.shape[0]
        nY = Y_data.shape[0]
        task_X = X[:, task_col]
        task_Y = Y[:, task_col]
        unique_X, groups_idx_X = npi.group_by(task_X,
                                              np.arange(nX))
        unique_Y, groups_idx_Y = npi.group_by(task_Y,
                                              np.arange(nY))
        A = np.zeros((nX, nY))
        
        for i, tx in enumerate(unique_X):
            for j, ty in enumerate(unique_Y):
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                if tx in order_delta:
                    order_tx = order_delta[tx]
                else:
                    order_tx = order_delta[float(tx)]
                if ty in order_delta:
                    order_ty = order_delta[ty]
                else:
                    order_ty = order_delta[float(ty)]
                a_xy = self.deltainv[order_tx, order_ty]
                A[indX[:, None], indY] = a_xy

        Q = self._apply_kernel(ckernel, X_data, Y_data, **dict(kwargs,
                               common=True))
        
        K = np.zeros((nX, nY))
        for i, tx in enumerate(unique_X):
            arr_j = np.where(unique_Y == tx)[0]
            if len(arr_j) == 1:
                j = arr_j[0]
                indX = groups_idx_X[i]
                indY = groups_idx_Y[j]
                K[indX[:, None], indY] = self._compute_K(X_data[indX],
                                                         Y_data[indY],
                                                         tx, tx,
                                                         skernel, **kwargs)

        hat_Q = self.mu**2 * np.multiply(A, Q) + (1-self.mu)**2 * K
        return hat_Q 


class MTLSVC(MTLSVM, ClassifierMixin):
    """docstring for MTLSVM."""
    def __init__(self, C=1.0, ckernel='rbf', skernel='rbf', degree=3,
                 cgamma='auto', sgamma='auto', coef0=0.0,
                 shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, task_info=None,
                 decision_function_shape='ovr', random_state=None, mu=1.0, mu2=1.0, mtl_type='convex',
                 delta=None, order_delta=None):
        super(MTLSVC, self).__init__(C, ckernel, skernel, degree, cgamma,
                                     sgamma, coef0,
                                     shrinking, tol, cache_size, verbose,
                                     max_iter, mu, mu2, task_info ,mtl_type, delta, order_delta)
        self.delta = delta
        self.order_delta = order_delta    
        self.probability = probability
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state
        self.score_fun = accuracy_score

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.svm = SVC(self.C, kernel, self.degree, gamma, self.coef0,
                       self.shrinking, self.probability, self.tol, self.cache_size,
                       self.class_weight, self.verbose, self.max_iter,
                       self.decision_function_shape, self.random_state)
        return super().fit(X, y, task_info, **kwargs)

    def decision_function(self, X):
        return self.svm.decision_function(X)


class MTLSVR(MTLSVM, RegressorMixin):
    """docstring for MTLSVM."""
    def __init__(self, ckernel='rbf', skernel='rbf', degree=3, cgamma='auto',
                 sgamma='auto', coef0=0.0, tol=0.001, C=1.0,
                 epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, mu=1.0, mu2=1.0, task_info=None, mtl_type='convex', delta=None, order_delta=None):
        super(MTLSVR, self).__init__(C, ckernel, skernel, degree, cgamma,
                                     sgamma, coef0, shrinking, tol, cache_size,
                                     verbose, max_iter, mu, mu2, task_info, mtl_type, delta, order_delta)
        self.epsilon = epsilon
        self.score_fun = r2_score

    def fit(self, X, y, task_info=-1, sample_weight=None, **kwargs):
        kernel = 'precomputed'  # We use the GRAMM matrix
        gamma = 'auto'
        self.svm = SVR(kernel, self.degree, gamma, self.coef0,
                       self.tol, self.C, self.epsilon, self.shrinking, self.cache_size,
                       self.verbose, self.max_iter)
        return super().fit(X, y, task_info, **kwargs)
