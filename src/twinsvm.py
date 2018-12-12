# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: twinsvm.py
In this module, functios is defined for training and testing TwinSVM classifier.

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
# ClippDCD optimizer is an extension module which is implemented in C++
import clippdcd


class TSVM:

    def __init__(self, kernel_type='linear', rect_kernel=1, c1=2**0, c2=2**0, \
                 gamma=2**0):

        """
            Input:
            Kernel_type: 1->Linear, 2->RBF(Gaussion)
            c1, c2: Penalty parameters
            gamma: RBF function parameter
        """

        self.C1 = c1
        self.C2 = c2
        self.u = gamma
        self.kernel_t = kernel_type
        self.rectangular_size = rect_kernel
        self.mat_C_t = None

        # Two hyperplanes attributes
        self.w1, self.b1, self.w2, self.b2 = None, None, None, None

    def set_parameter(self, c1=2**0, c2=2**0, gamma=2**0):

        """
        It changes the parametes for TSVM classifier.
        DO NOT USE THIS METHOD AFTER INSTANTIATION OF TSVM CLASS!
        THIS METHOD CREATED ONLY FOR Validator CLASS.
        Input:
            c1, c2: Penalty parameters
            gamma: RBF function parameter
        """

        self.C1 = c1
        self.C2 = c2
        self.u = gamma

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

        if self.kernel_t == 'linear':  # Linear kernel
            
            mat_H = np.column_stack((mat_A, mat_e1))
            mat_G = np.column_stack((mat_B, mat_e2))

        elif self.kernel_t == 'RBF': # Non-linear 

            # class 1 & class -1
            mat_C = np.row_stack((mat_A, mat_B))

            self.mat_C_t = np.transpose(mat_C)[:, :int(mat_C.shape[0] * self.rectangular_size)]

            mat_H = np.column_stack((rbf_kernel(mat_A, self.mat_C_t, self.u), mat_e1))

            mat_G = np.column_stack((rbf_kernel(mat_B, self.mat_C_t, self.u), mat_e2))


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

        # Obtaining Lagrane multipliers using ClippDCD optimizer
        alpha_d1 = np.array(clippdcd.clippDCD_optimizer(mat_dual1, self.C1)).reshape(mat_dual1.shape[0], 1)
        alpha_d2 = np.array(clippdcd.clippDCD_optimizer(mat_dual2, self.C2)).reshape(mat_dual2.shape[0], 1)

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
                    self.mat_C_t, self.u)}

        for i in range(X_test.shape[0]):

            # Prependicular distance of data pint i from hyperplanes
            prepen_distance[i, 1] = np.abs(np.dot(kernel_f[self.kernel_t](i), self.w1) + self.b1)

            prepen_distance[i, 0] = np.abs(np.dot(kernel_f[self.kernel_t](i), self.w2) + self.b2)

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


class MCTSVM:

    """
    Multi Class Twin Support Vector Machine
    One-vs-All Scheme
    """

    def __init__(self, kernel_type='linear', c=2**0, gamma=2**0):

        """
        Input:
            kernel_type: Type of kernel function (Linear or RBF)
            c: Penalty parameter
            gamma: parameter of RBF function
        """

        self.kernel_t = kernel_type
        self.C = c
        self.y = gamma
        self.classfiers = {}  # Classifiers
        self.mat_D_t = []  # For non-linear MCTSVM

    def set_parameter(self, c=2**0, gamma=2**0):

        """
        It changes the parametes for multiclass TSVM classifier.
        DO NOT USE THIS METHOD AFTER INSTANTIATION OF MCTSVM CLASS!
        THIS METHOD CREATED ONLY FOR Validator CLASS.
        Input:
            c: Penalty parameters
            gamma: RBF function parameter
        """

        self.C = c
        self.y = gamma

    def fit(self, X_train, y_train):

        """
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

            if self.kernel_t == 'linear':
                
                mat_A_i = np.column_stack((mat_X_i, mat_e1_i))
                mat_B_i = np.column_stack((mat_Y_i, mat_e2_i))

            elif self.kernel_t == 'RBF':

                mat_D = np.row_stack((mat_X_i, mat_Y_i))

                self.mat_D_t.append(np.transpose(mat_D))

                mat_A_i = np.column_stack((rbf_kernel(mat_X_i, self.mat_D_t[idx], self.y), mat_e1_i))
                mat_B_i = np.column_stack((rbf_kernel(mat_Y_i, self.mat_D_t[idx], self.y), mat_e2_i))

            mat_A_i_t = np.transpose(mat_A_i)
            mat_B_i_t = np.transpose(mat_B_i)

            # Compute inverses:
            # Regulariztion term used for ill-possible condition
            reg_term = 2 ** float(-7)
    
            mat_A_A = np.linalg.inv(np.dot(mat_A_i_t, mat_A_i) + (reg_term * np.identity(mat_A_i.shape[1])))
    
            # Dual problem of i-th class
            mat_dual_i = np.dot(np.dot(mat_B_i, mat_A_A), mat_B_i_t)
    
            # Obtaining Lagrange multipliers using ClippDCD optimizer
            alpha_i = np.array(clippdcd.clippDCD_optimizer(mat_dual_i, self.C)).reshape(mat_dual_i.shape[0], 1)
    
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
                    self.mat_D_t[j], self.y)}

        for i in range(X_test.shape[0]):

            for idx, j in enumerate(self.classfiers.keys()):

                prepen_dist[i, idx] = np.abs(np.dot(kernel_f[self.kernel_t](i, idx), \
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
    
    def __init__(self, kernel='linear', C_1=1, C_2=1, gamma=1):
        
        """
        Parameters:
            kernel: Type of kernel function. 'linear' or 'RBF'
            C_1, C_2: float, (default=1)
                      Penalty parameter
            gamma: float, (default=1.0)
                   Kernel coefficient for RBF function
                
        """
        
        self.kernel = kernel
        self.C_1 = C_1
        self.C_2 = C_2
        self.gamma = gamma
         
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
                
                self.bin_TSVM_models_[p] = TSVM(self.kernel, 1, self.C_1, self.C_2, \
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
    
    from dataproc import read_data
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.model_selection import cross_val_score, GridSearchCV
    import time

    
    train_data, labels, data_name = read_data('/home/mir/mir-projects/Mir-Repo/dataset/mc-data/wine.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels,
                                                        test_size=0.30, random_state=42)
    
    
#    param = {'C_1': [float(2**i) for i in range(-5, 6)],
#             'C_2': [float(2**i) for i in range(-5, 6)]}
    
    start_t = time.time()
#    
    ovo_tsvm_model = OVO_TSVM('linear')
    
    #cv = cross_val_score(ovo_tsvm_model, train_data, labels, cv=10)
    
#    result = GridSearchCV(ovo_tsvm_model, param, cv=10, n_jobs=4, refit=False, verbose=1)
#    result.fit(train_data, labels)
    
#    
    ovo_tsvm_model.fit(X_train, y_train)
    
    pred = ovo_tsvm_model.predict(X_test)
#    
    print("Finished: %.2f ms" % ((time.time() - start_t) * 1000))
#    
    print("Accuracy: %.2f" % (accuracy_score(y_test, pred) * 100))
