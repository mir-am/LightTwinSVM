# -*- coding: utf-8 -*-

# LightTwinSVM Program - Simple and Fast
# Version: 0.5.0 - 2019-01-01
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

"""
Classes and functios are defined for training and testing TwinSVM classifier.

TwinSVM classifier generates two non-parallel hyperplanes.
For more info, refer to the original papar.
Khemchandani, R., & Chandra, S. (2007). Twin support vector machines for pattern classification. IEEE Transactions on pattern analysis and machine intelligence, 29(5), 905-910.

Motivated by the following paper, the multi-class TSVM is developed.
Tomar, D., & Agarwal, S. (2015). A comparison on multi-class classification methods based on least squares twin support vector machine. Knowledge-Based Systems, 81, 131-147.
"""


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import column_or_1d
import numpy as np

# ClipDCD optimizer is an extension module which is implemented in C++
from ltsvm.optimizer import clipdcd


class TSVM(BaseEstimator):

    def __init__(self, kernel='linear', rect_kernel=1, C1=2**0, C2=2**0, \
                 gamma=2**0):

        """
        Input:
            Kernel_type: 1->Linear, 2->RBF(Gaussion)
            c1, c2: Penalty parameters
            gamma: RBF function parameter
        """

        self.C1 = C1
        self.C2 = C2
        self.gamma = gamma
        self.kernel = kernel
        self.rect_kernel = rect_kernel
        self.mat_C_t = None
        self.cls_name = 'TSVM'
        
        # Two hyperplanes attributes
        self.w1, self.b1, self.w2, self.b2 = None, None, None, None

    def get_params_names(self):
        
        """
        It returns the names of hyper-parameters of this classifier.
        """
        
        return ['C1', 'C2'] if self.kernel == 'linear' else ['C1', 'C2', 'gamma']

    def fit(self, X_train, y_train):

        """
        It trains TwinSVM classfier on given data
        Input:    
            X_train: Training samples
            y_train: Samples' category
        output:

            w1, w2: Coordinates of two non-parallel hyperplanes
            b1, b2: Biases
            """

        # Matrix A or class 1 samples
        mat_A = X_train[y_train == 1]

        # Matrix B  or class -1 data 
        mat_B = X_train[y_train == -1]

        # Vectors of ones
        mat_e1 = np.ones((mat_A.shape[0], 1))
        mat_e2 = np.ones((mat_B.shape[0], 1))

        if self.kernel == 'linear':  # Linear kernel
            
            mat_H = np.column_stack((mat_A, mat_e1))
            mat_G = np.column_stack((mat_B, mat_e2))

        elif self.kernel == 'RBF': # Non-linear 

            # class 1 & class -1
            mat_C = np.row_stack((mat_A, mat_B))

            self.mat_C_t = np.transpose(mat_C)[:, :int(mat_C.shape[0] * self.rect_kernel)]

            mat_H = np.column_stack((rbf_kernel(mat_A, self.mat_C_t, self.gamma), mat_e1))

            mat_G = np.column_stack((rbf_kernel(mat_B, self.mat_C_t, self.gamma), mat_e2))


        mat_H_t = np.transpose(mat_H)
        mat_G_t = np.transpose(mat_G)

        # Compute inverses:
        # Regulariztion term used for ill-possible condition
        reg_term = 2 ** float(-7)

        mat_H_H = np.linalg.inv(np.dot(mat_H_t, mat_H) + (reg_term * np.identity(mat_H.shape[1])))
        mat_G_G = np.linalg.inv(np.dot(mat_G_t, mat_G) + (reg_term * np.identity(mat_G.shape[1])))

        # Wolfe dual problem of class 1
        mat_dual1 = np.dot(np.dot(mat_G, mat_H_H), mat_G_t)
        # Wolfe dual problem of class -1
        mat_dual2 = np.dot(np.dot(mat_H, mat_G_G), mat_H_t)

        # Obtaining Lagrange multipliers using ClipDCD optimizer
        alpha_d1 = np.array(clipdcd.clippDCD_optimizer(mat_dual1, self.C1)).reshape(mat_dual1.shape[0], 1)
        alpha_d2 = np.array(clipdcd.clippDCD_optimizer(mat_dual2, self.C2)).reshape(mat_dual2.shape[0], 1)

        # Obtain hyperplanes
        hyper_p_1 = -1 * np.dot(np.dot(mat_H_H, mat_G_t), alpha_d1)

        # Class 1
        self.w1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
        self.b1 = hyper_p_1[-1, :]

        hyper_p_2 = np.dot(np.dot(mat_G_G, mat_H_t), alpha_d2)

        # Class -1
        self.w2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
        self.b2 = hyper_p_2[-1, :]


    def predict(self, X_test):

        """
            Predictes class of test samples
            Input:
                X_test: Test samples    
        """

        # Calculate prependicular distances for new data points 
        prepen_distance = np.zeros((X_test.shape[0], 2))

        kernel_f = {'linear': lambda i: X_test[i, :] , 'RBF': lambda i: rbf_kernel(X_test[i, :], \
                    self.mat_C_t, self.gamma)}

        for i in range(X_test.shape[0]):

            # Prependicular distance of data pint i from hyperplanes
            prepen_distance[i, 1] = np.abs(np.dot(kernel_f[self.kernel](i), self.w1) + self.b1)

            prepen_distance[i, 0] = np.abs(np.dot(kernel_f[self.kernel](i), self.w2) + self.b2)

        # Assign data points to class +1 or -1 based on distance from hyperplanes
        output = 2 * np.argmin(prepen_distance, axis=1) - 1

        return output


def rbf_kernel(x, y, u):

    """
        It transforms samples into higher dimension  
        Input:
            x,y: Samples
            u: Gamma parameter      
        Output:
            Samples with higher dimension
    """

    return np.exp(-2 * u) * np.exp(2 * u * np.dot(x, y))


class HyperPlane:

    def __init__(self):

        self.w = None  # Coordinates of hyperplane
        self.b = None  # Bias term


class MCTSVM(BaseEstimator):

    """
    Multi Class Twin Support Vector Machine
    One-vs-All Scheme
    """

    def __init__(self, kernel='linear', C=2**0, gamma=2**0):

        """
        Input:
            kernel_type: Type of kernel function (Linear or RBF)
            c: Penalty parameter
            gamma: parameter of RBF function
        """

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.classfiers = {}  # Classifiers
        self.mat_D_t = []  # For non-linear MCTSVM
        self.cls_name = 'TSVM_OVA'

    def get_params_names(self):
        
        """
        It returns the names of hyper-parameters of this classifier.
        """
        
        return ['C'] if self.kernel == 'linear' else ['C', 'gamma']

    def fit(self, X_train, y_train):

        """
        Input:
            X_train: Training samples
            y_train: Lables of training samples
        """

        num_classes = np.unique(y_train)

        # Construct K-binary classifiers
        for idx, i in enumerate(num_classes):

            # Samples of i-th class
            mat_X_i = X_train[y_train == i]

            # Samples of other classes
            mat_Y_i = X_train[y_train != i]

            # Vectors of ones
            mat_e1_i = np.ones((mat_X_i.shape[0], 1))
            mat_e2_i = np.ones((mat_Y_i.shape[0], 1))

            if self.kernel == 'linear':
                
                mat_A_i = np.column_stack((mat_X_i, mat_e1_i))
                mat_B_i = np.column_stack((mat_Y_i, mat_e2_i))

            elif self.kernel == 'RBF':

                mat_D = np.row_stack((mat_X_i, mat_Y_i))

                self.mat_D_t.append(np.transpose(mat_D))

                mat_A_i = np.column_stack((rbf_kernel(mat_X_i, self.mat_D_t[idx], self.gamma), mat_e1_i))
                mat_B_i = np.column_stack((rbf_kernel(mat_Y_i, self.mat_D_t[idx], self.gamma), mat_e2_i))

            mat_A_i_t = np.transpose(mat_A_i)
            mat_B_i_t = np.transpose(mat_B_i)

            # Compute inverses:
            # Regulariztion term used for ill-possible condition
            reg_term = 2 ** float(-7)
    
            mat_A_A = np.linalg.inv(np.dot(mat_A_i_t, mat_A_i) + (reg_term * np.identity(mat_A_i.shape[1])))
    
            # Dual problem of i-th class
            mat_dual_i = np.dot(np.dot(mat_B_i, mat_A_A), mat_B_i_t)
    
            # Obtaining Lagrange multipliers using ClippDCD optimizer
            alpha_i = np.array(clipdcd.clippDCD_optimizer(mat_dual_i, self.C)).reshape(mat_dual_i.shape[0], 1)
    
            hyperplane_i = np.dot(np.dot(mat_A_A, mat_B_i_t), alpha_i)
    
            hyper_p_inst = HyperPlane()
            hyper_p_inst.w = hyperplane_i[:hyperplane_i.shape[0] - 1, :]
            hyper_p_inst.b = hyperplane_i[-1, :]
    
            self.classfiers[i] = hyper_p_inst


    def predict(self, X_test):

        """
        Predictes class of test samples
            Input:
                X_test: Test samples
        """

        # Perpendicular distance from each hyperplane
        prepen_dist = np.zeros((X_test.shape[0], len(self.classfiers.keys())))

        kernel_f = {'linear': lambda i, j: X_test[i, :] , 'RBF': lambda i, j: rbf_kernel(X_test[i, :], \
                    self.mat_D_t[j], self.gamma)}

        for i in range(X_test.shape[0]):

            for idx, j in enumerate(self.classfiers.keys()):

                prepen_dist[i, idx] = np.abs(np.dot(kernel_f[self.kernel](i, idx), \
                           self.classfiers[j].w) + self.classfiers[j].b) / np.linalg.norm(self.classfiers[j].w)

        output = np.argmin(prepen_dist, axis=1) + 1

        return output


class OVO_TSVM(BaseEstimator, ClassifierMixin):

    """
    Multi Class Twin Support Vector Machine
    One-vs-One Scheme
    This classifier is scikit-learn compatible, which means scikit-learn features
    such as cross_val_score and GridSearchCV can be used for OVO_TSVM
    """    
    
    def __init__(self, kernel='linear', C1=1, C2=1, gamma=1):
        
        """
        Parameters:
            kernel: Type of kernel function. 'linear' or 'RBF'
            C_1, C_2: float, (default=1)
                      Penalty parameter
            gamma: float, (default=1.0)
                   Kernel coefficient for RBF function
                
        """
        
        self.kernel = kernel
        self.C1 = C1
        self.C2 = C2
        self.gamma = gamma
        self.cls_name = 'TSVM_OVO'
        
    def get_params_names(self):
        
        """
        It returns the names of hyper-parameters of this classifier.
        """
        
        return ['C1', 'C2'] if self.kernel == 'linear' else ['C1', 'C2', 'gamma']
         
    def _validate_targets(self, y):
        
        """
        Validates labels for training and testing classifier
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y_, return_inverse=True)
        
        return np.asarray(y, dtype=np.int)
    
    
    def _validate_for_predict(self, X):
        
        """
        Checks that the classifier is already trained and also test samples are
        valid
        """
        
        check_is_fitted(self, ['bin_TSVM_models_'])
        X = check_array(X, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        if n_features != self.shape_fit_[1]:
            
            raise ValueError("X.shape[1] = %d should be equal to %d," 
                             "the number of features of training samples" % 
                             (n_features, self.shape_fit_[1]))
        
        return X
    
    
    def fit(self, X, y):
        
        
        """
        Given training set, it creates a SVM model
        
        Parameters:
            X_train: Training samples, (n_samples, n_features)
            y_train: Target values, (n_samples, )
        """
        
        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64)
         
        # Allocate n(n-1)/2 binary TSVM classifiers
        self.bin_TSVM_models_ = ((self.classes_.size * (self.classes_.size - 1))
                               // 2 ) * [None]
        
        p = 0
        
        for i in range(self.classes_.size):
            
            for j in range(i + 1, self.classes_.size):
                
                #print("%d, %d" % (i, j))
                
                # Break multi-class problem into a binary problem
                sub_prob_X_i_j = X[(y == i) | (y == j)]
                sub_prob_y_i_j = y[(y == i) | (y == j)]
                
                #print(sub_prob_y_i_j)
                
                # For binary classification, labels must be {-1, +1}
                # i-th class -> +1 and j-th class -> -1
                sub_prob_y_i_j[sub_prob_y_i_j == j] = -1
                sub_prob_y_i_j[sub_prob_y_i_j == i] = 1
                
                self.bin_TSVM_models_[p] = TSVM(self.kernel, 1, self.C1, self.C2, \
                               self.gamma)
                
                self.bin_TSVM_models_[p].fit(sub_prob_X_i_j, sub_prob_y_i_j)
                
                p = p + 1
                
        self.shape_fit_ = X.shape
                
        return self
         
    def predict(self, X):
        
        """
        Predicits lables of test samples
        
        Parameters:
            X_test: test samples, (n_samples, n_features)
        
        Returns:
            y_pred: array, (n_samples,)
        
        """
        
        X = self._validate_for_predict(X)
        
        # Initialze votes
        votes = np.zeros((X.shape[0], self.classes_.size), dtype=np.int)
        
        # iterate over test samples
        for k in range(X.shape[0]):
            
            p = 0
        
            for i in range(self.classes_.size):
                
                for j in range(i + 1, self.classes_.size):
                    
                    y_pred = self.bin_TSVM_models_[p].predict(X[k, :].reshape(1, X.shape[1]))
                    
                    if y_pred == 1:
                        
                        votes[k, i] = votes[k, i] + 1
                        
                    else:
                        
                        votes[k, j] = votes[k, j] + 1
                        
                    p = p + 1
                        
        
         # Labels of test samples based max-win strategy
        max_votes = np.argmax(votes, axis=1)
            
        return self.classes_.take(np.asarray(max_votes, dtype=np.int))
                
        
        
    
if __name__ == '__main__':
    
#    from ltsvm.dataproc import read_data
#    from sklearn.metrics import accuracy_score
#    from sklearn.model_selection import train_test_split
    from sklearn.utils.estimator_checks import check_estimator
#    from sklearn.model_selection import cross_val_score, GridSearchCV
#    import time
#    
    check_estimator(OVO_TSVM)

    
#    train_data, labels, data_name = read_data('/home/mir/mir-projects/Mir-Repo/mc-data/wine.csv')
#    
#    X_train, X_test, y_train, y_test = train_test_split(train_data, labels,
#                                                        test_size=0.30, random_state=42)
#    
#    
##    param = {'C_1': [float(2**i) for i in range(-5, 6)],
##             'C_2': [float(2**i) for i in range(-5, 6)]}
#    
#    start_t = time.time()
##    
#    ovo_tsvm_model = MCTSVM()
#    ovo_tsvm_model.set_params(**{'C': 4, 'gamma': 0.1})
#    print(ovo_tsvm_model.get_params())
#    
#    #cv = cross_val_score(ovo_tsvm_model, train_data, labels, cv=10)
#    
##    result = GridSearchCV(ovo_tsvm_model, param, cv=10, n_jobs=4, refit=False, verbose=1)
##    result.fit(train_data, labels)
#    
#    print(X_train.shape)
##    
#    ovo_tsvm_model.fit(X_train, y_train)
#    
#    pred = ovo_tsvm_model.predict(X_test)
##    
#    print("Finished: %.2f ms" % ((time.time() - start_t) * 1000))
##    
#    print("Accuracy: %.2f" % (accuracy_score(y_test, pred) * 100))