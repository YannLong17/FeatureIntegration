from sklearn.naive_bayes import BaseNB, check_X_y, check_array
import scipy.optimize as opt
import numpy as np

class PoissonNB(BaseNB):
    """
    Attributes
    --------
    class_prior_ : array, shape(n_class,)
        Probability of each class

    class_count_ : array, shape(n_class,)
        Number of training Sample in each class

    lambda_ : array, shape(n_class, n_features)
        mean of each feature per class

    """

    def fit(self, X, y):
        """
        :param X: Design Matrix, shape(n_trials,n_features)
        :param y: Response Vector, shape(n_trials,)
        :return: self
        """

        X, y = check_X_y(X, y)
        PoissonNB.check_non_negative(X)

        n_trials, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        epsilon = 1e-9

        self.lambda_ = np.zeros((n_classes, n_features))

        self.class_prior_ = np.zeros(n_classes)

        for i, y_i in enumerate(unique_y):
            Xi = X[y == y_i, :]
            self.lambda_[i, :] = np.mean(Xi, axis=0) + epsilon
            self.class_prior_[i] = float(Xi.shape[0])/n_trials

        return self

    def _joint_log_likelihood(self, X):
        X = check_array(X)
        PoissonNB.check_non_negative(X)

        joint_log_likelihood = np.zeros((np.shape(X)[0],len(self.classes_)))

        for i in range(len(self.classes_)):
            n_ij = np.sum(X * np.log(self.lambda_[i, :]), axis=1)
            n_ij -= np.sum(self.lambda_[i, :])
            joint_log_likelihood[:, i] = np.log(self.class_prior_[i]) + n_ij

        return joint_log_likelihood

    @staticmethod
    def check_non_negative(X):
        if np.any(X < 0.):
            raise ValueError("Input X must be non-negative")


class ZipNB(BaseNB):
    """
    Zero Inflated Poisson Naive Bayes
    Attributes
    --------
    class_prior_ : array, shape(n_class,)
        Probability of each class

    class_count_ : array, shape(n_class,)
        Number of training Sample in each class

    lambda_ : array, shape(n_class, n_features)
        mean of the poisson componert for each feature per class

    pi_ : array ,shape(n_class, n_features)
        proportion of zero component for each feature per class

    """

    def __init__(self, estimator='moment'):
        if estimator not in ['MLE', 'moment']:
            raise SyntaxError('Estimator must be \'MLE\' or \'moment\' (but MLE is still broken at the moment)' )
        self.estimator = estimator

    def fit(self, X, y):
        """
        :param X: Design Matrix, shape(n_trials,n_features)
        :param y: Response Vector, shape(n_trials,)
        :return: self
        """

        X, y = check_X_y(X, y)
        ZipNB.check_non_negative(X)

        n_trials, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        epsilon = 1e-9

        self.lambda_ = np.zeros((n_classes, n_features))

        self.pi_ = np.zeros((n_classes, n_features))

        self.class_prior_ = np.zeros(n_classes)

        if self.estimator == 'moment':
            for i, y_i in enumerate(unique_y):
                Xi = X[y == y_i, :]
                m = np.mean(Xi, axis=0) + epsilon
                s = np.var(Xi, axis=0) + epsilon

                self.lambda_[i, :] = m + s/m - 1

                if np.amin(self.lambda_)<0:
                    print("m=%f s=%f " %(m[4],s[4]))
                self.pi_[i, :] = np.divide((s-m), (s+np.square(m)-m))
                self.class_prior_[i] = float(Xi.shape[0]) / n_trials

        elif self.estimator == 'MLE':
            for i, y_i in enumerate(unique_y):
                Xi = X[y == y_i, :]
                m = np.mean(Xi, axis=0) + epsilon
                n = Xi.shape[0]
                n0 = (Xi == 0).sum(0)/Xi.shape[0]
                self.lambda_[i, :] = opt.fsolve(func=ZipNB.fun, x0=m, args=(m))
                self.lambda_[self.lambda_[i, :] < 0 ] = epsilon
                self.pi_[i, :] = 1 - m/self.lambda_[i, :]
                self.class_prior_[i] = float(Xi.shape[0]) / n_trials

        return self


    def _joint_log_likelihood(self, X):
        X = check_array(X)
        PoissonNB.check_non_negative(X)

        joint_log_likelihood = np.zeros((np.shape(X)[0], len(self.classes_)))

        for i in range(len(self.classes_)):
            # zero component
            n_ij = np.sum((X[...] == 0)*np.log(self.pi_[i, :] + (1-self.pi_[i, :])*np.exp(-self.lambda_[i, :])), axis=1)
            # Non Zero component
            n_ij += np.sum(X * np.log(self.lambda_[i, :]), axis=1)
            n_ij += np.sum((X[...] != 0)*(np.log(1-self.pi_[i, :]) - np.log(self.lambda_[i, :] )), axis=1)
            joint_log_likelihood[:, i] = np.log(self.class_prior_[i]) + n_ij

        return joint_log_likelihood

    @staticmethod
    def check_non_negative(X):
        if np.any(X < 0.):
            raise ValueError("Input X must be non-negative")

    @staticmethod
    def fun(x, m):
        return m*(1 - np.exp(-x)) - x


class ExpNB(BaseNB):
    """
    Exponential Naive Bayes
    Attributes
    --------
    class_prior_ : array, shape(n_class,)
        Probability of each class

    class_count_ : array, shape(n_class,)
        Number of training Sample in each class

    lambda_ : array, shape(n_class, n_features)
        mean of each feature per class

    """

    def fit(self, X, y):
        """
        :param X: Design Matrix, shape(n_trials,n_features)
        :param y: Response Vector, shape(n_trials,)
        :return: self
        """

        X, y = check_X_y(X, y)
        ZipNB.check_non_negative(X)

        n_trials, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]

        epsilon = 1e-9

        self.lambda_ = np.zeros((n_classes, n_features))

        self.class_prior_ = np.zeros(n_classes)

        for i, y_i in enumerate(unique_y):
            Xi = X[y == y_i, :]
            self.lambda_[i, :] = np.mean(Xi, axis=0) + epsilon
            self.class_prior_[i] = float(Xi.shape[0]) / n_trials

        return self

    def _joint_log_likelihood(self, X):
        X = check_array(X)
        ExpNB.check_non_negative(X)

        joint_log_likelihood = np.zeros((np.shape(X)[0], len(self.classes_)))

        for i in range(len(self.classes_)):
            n_ij = np.sum(np.log(self.lambda_[i, :]) - X*self.lambda_[i, :], axis=1)
            joint_log_likelihood[:, i] = np.log(self.class_prior_[i]) + n_ij

        return joint_log_likelihood

    @staticmethod
    def check_non_negative(X):
        if np.any(X < 0.):
            raise ValueError("Input X must be non-negative")
