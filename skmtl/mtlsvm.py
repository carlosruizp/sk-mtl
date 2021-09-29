"""
This is a module to develop the classes of Multi-Task Learning SVMs
"""
import warnings

import numpy as np
from scipy.sparse import isspmatrix

import numpy_indexed as npi

from sklearn.svm import SVC
from sklearn.svm import SVR
import sklearn.metrics.pairwise as pairwise
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, r2_score

from icecream import ic

MTL_TYPES = ['joint', 'laplacian']

class ConvexMTLSVC(SVC):
    """ A class for Multi-Task SVM classificators.

    TODO
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.
    degree : int, default=3
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        - if ``gamma='scale'`` (default) is passed then it uses
          1 / (n_features * X.var()) as value of gamma,
        - if 'auto', uses 1 / n_features.
        .. versionchanged:: 0.22
           The default value of ``gamma`` changed from 'auto' to 'scale'.
    coef0 : float, default=0.0
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.
    shrinking : bool, default=True
        Whether to use the shrinking heuristic.
        See the :ref:`User Guide <shrinking_svm>`.
    probability : bool, default=False
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, will slow down that method as it internally uses
        5-fold cross-validation, and `predict_proba` may be inconsistent with
        `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    cache_size : float, default=200
        Specify the size of the kernel cache (in MB).
    class_weight : dict or 'balanced', default=None
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
    verbose : bool, default=False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.
    max_iter : int, default=-1
        Hard limit on iterations within solver, or -1 for no limit.
    decision_function_shape : {'ovo', 'ovr'}, default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2). However, one-vs-one
        ('ovo') is always used as multi-class strategy. The parameter is
        ignored for binary classification.
        .. versionchanged:: 0.19
            decision_function_shape is 'ovr' by default.
        .. versionadded:: 0.17
           *decision_function_shape='ovr'* is recommended.
        .. versionchanged:: 0.17
           Deprecated *decision_function_shape='ovo' and None*.
    break_ties : bool, default=False
        If true, ``decision_function_shape='ovr'``, and number of classes > 2,
        :term:`predict` will break ties according to the confidence values of
        :term:`decision_function`; otherwise the first class among the tied
        classes is returned. Please note that breaking ties comes at a
        relatively high computational cost compared to a simple predict.
        .. versionadded:: 0.22
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generation for shuffling the data for
        probability estimates. Ignored when `probability` is False.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    mtl_type: str, default='joint'
        Defines the choice of the type of MTL approach used.
        - If 'joint' a joint learning of a common and task-speficic models
        is used.
        - If 'laplacian' a laplacian regularization approach is used. 
    Attributes
    ----------
    class_weight_ : ndarray of shape (n_classes,)
        Multipliers of parameter C for each class.
        Computed based on the ``class_weight`` parameter.
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.
        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.
    dual_coef_ : ndarray of shape (n_classes -1, n_SV)
        Dual coefficients of the support vector in the decision
        function (see :ref:`sgd_mathematical_formulation`), multiplied by
        their targets.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the :ref:`multi-class section of the User Guide
        <svm_multi_class>` for details.
    fit_status_ : int
        0 if correctly fitted, 1 otherwise (will raise warning)
    intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)
        Constants in decision function.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    support_ : ndarray of shape (n_SV)
        Indices of support vectors.
    support_vectors_ : ndarray of shape (n_SV, n_features)
        Support vectors.
    n_support_ : ndarray of shape (n_classes,), dtype=int32
        Number of support vectors for each class.
    probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
    probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)
        If `probability=True`, it corresponds to the parameters learned in
        Platt scaling to produce probability estimates from decision values.
        If `probability=False`, it's an empty array. Platt scaling uses the
        logistic function
        ``1 / (1 + exp(decision_value * probA_ + probB_))``
        where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
        more information on the multiclass case and training procedure see
        section 8 of [1]_.
    shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
        Array dimensions of training vector ``X``.
    See Also
    --------
    SVR : Support Vector Machine for Regression implemented using libsvm.
    LinearSVC : Scalable Linear Support Vector Machine for classification
        implemented using liblinear. Check the See Also section of
        LinearSVC for more comparison element.
    References
    ----------
    .. [1] `LIBSVM: A Library for Support Vector Machines
        <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_
    .. [2] `Platt, John (1999). "Probabilistic outputs for support vector
        machines and comparison to regularizedlikelihood methods."
        <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> X = np.ndarray([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    >>> y = np.ndarray([1, 1, 2, 2])
    >>> from sklearn.svm import SVC
    >>> clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    >>> clf.fit(X, y)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])
    >>> print(clf.predict([[-0.8, -1]]))
    [1]

    """
    def __init__(
        self,
        *,
        C=1.0,
        ckernel="rbf",
        skernel="rbf",
        cdegree=3,
        sdegree=3,
        cgamma="scale",
        sgamma="scale",
        ccoef0=0.0,
        scoef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
        lamb=0.5,
        mtl_type='joint'
    ):

        super().__init__(
            C=C,
            kernel='precomputed', # the kernel matrix is computed in the fit methods
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state
        )
        self.ckernel = ckernel
        self.skernel = skernel
        self.cdegree = cdegree
        self.sdegree = sdegree
        self.cgamma = cgamma
        self.sgamma = sgamma
        self.ccoef0 = ccoef0
        self.scoef0 = scoef0
        self.lamb = lamb
        self.mtl_type = mtl_type

    def fit(self, X, y, task_info=-1, sample_weight=None):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        # input validation
        X, y = check_X_y(X, y, accept_sparse=True)
        n_samples, n_dim = X.shape

        if isinstance(task_info, int):
            if task_info >= n_dim:
                raise ValueError(
                    "task_info should be between 0"
                    "and {} or -1, {} was given".format(n_dim - 1, task_info)
                                 )
            else:
                X_task = X[task_info].flatten()
                X_full = X
                X = np.delete(X, task_info, axis=1)
        elif isinstance(task_info, np.ndarray):
            if len(task_info.flatten()) != n_samples:
                raise ValueError(
                    "The task column and the data must have the same number of instances."
                    "{} and {} sizes were given".format(len(task_info.flatten()), n)
                    )
            X_task = task_info.flatten()
            X_full = np.concatenate((X, X_task))
        else:
            raise TypeError('task_info must be an int or a numpy array')
       
        sparse = isspmatrix(X)
        self._sparse = sparse and not callable(self.kernel)

        # common kernel validation
        self._check_kernel(self.ckernel, 'ckernel')

        # common gamma validation
        self._cgamma = self._check_gamma(self.cgamma, 'cgamma', X, sparse)

        # common coef validation
        self._coef0 = self._check_coef0(self.ccoef0, 'ccoef0')        

        # common degree validation
        self._degree = self._check_degree(self.cdegree, 'cdegree')

        # specific kernel validation
        if isinstance(self.skernel, (list, np.ndarray)):
            if not [callable(k) for k in self.kernel].all():
                raise ValueError(
                    "If skernel is a list or numpy array, all members of skernel must be callable"
                    )
        elif isinstance(self.skernel, dict):
            if not [callable(k) for k in self.kernel.values()].all():
                raise ValueError(
                    "If skernel is a dict, all values of skernel must be callable"
                    )       
        else:
            self._check_kernel(self.skernel, 'skernel')
        
        # specific gamma validation
        if isinstance(self.sgamma, (list, np.ndarray)):
            self._sgamma = []
            for i, _gamma in enumerate(self.sgamma):
                self._sgamma.append(self._check_gamma(_gamma, 'sgamma_{}'.format(i)), X, sparse)
        elif isinstance(self.sgamma, dict):
            self._sgamma = {}
            for k, _gamma in self.sgamma.items():
                self._sgamma[k] = self._check_gamma(_gamma, 'sgamma_{}'.format(k), X, sparse)
        else:
            self._check_gamma(self.sgamma, 'sgamma', X, sparse)

        # specific coef0 validation
        if isinstance(self.scoef0, (list, np.ndarray)):
            self._scoef0 = []
            for i, _coef0 in enumerate(self.scoef0):
                self._scoef0.append(self._check_coef0(_coef0, 'scoef0_{}'.format(i)))
        elif isinstance(self.scoef0, dict):
            self._scoef0 = {}
            for k, _coef0 in self.scoef0.items():
                self._scoef0[k] = self._check_coef0(_coef0, 'scoef0_{}'.format(k))
        else:
            self._check_coef0(self.scoef0, 'scoef0')

        # specific degree validation
        if isinstance(self.sdegree, (list, np.ndarray)):
            self._sdegree = []
            for i, _degree in enumerate(self.sdegree):
                self._sdegree.append(self._check_degree(_degree, 'sdegree_{}'.format(i)))
        elif isinstance(self.sdegree, dict):
            self._sdegree = {}
            for k, _degree in self.sdegree.items():
                self._sdegree[k] = self._check_degree(_degree, 'sdegree_{}'.format(k))
        else:
            self._check_degree(self.sdegree, 'sdegree')

        ic(self.kernel)


        # Compute MTL kernel
        G = self._compute_mtl_kernel(X, X, mtl_type=self.mtl_type)

        # Fit
        super().fit(G, y, sample_weight=sample_weight)
        self.is_fitted_ = True

        # `fit` should always return `self`
        return self

    def _check_kernel(self, kernel, name):
        "Internal method to validate kernel values"
        if callable(kernel):
            pass
        elif isinstance(kernel, str):
            if kernel not in ['linear', 'rbf', 'poly', 'sigmoid']:
                raise ValueError(
                    "{} must be linear, rbf, sigmoid or poly".format(name)
                    )
        else:
            raise TypeError(
                "{} must be a callable object or str".format(name)
                )

    def _check_gamma(self, gamma, name, X, sparse):
        "Internal method to validate gamma values"
        if isinstance(gamma, str):
            if gamma == "scale":
                # var = E[X^2] - E[X]^2 if sparse
                X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
                return 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
            elif gamma == "auto":
                return 1.0 / X.shape[1]
            else:
                raise ValueError(
                    "When '{}' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(name, gamma)
                )
        else:
            return gamma

    def _check_coef0(self, coef0, name):
        "Internal method to validate coef0 values"
        if not isinstance(coef0, float):
            raise TypeError("{} must be an instance of float".format(name))
        return coef0

    def _check_degree(self, degree, name):
        "Internal method to validate degree values"
        if not isinstance(degree, int):
            raise TypeError("{} must be an instance of int".format(name))
        return degree

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)

    