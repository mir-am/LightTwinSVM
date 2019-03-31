#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LightTwinSVM Program - Simple and Fast
# Version: 0.6.0 - 2019-03-31
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

"""
In this module, classes and methods are defined for evluating the performance
of the TwinSVM model. Also, a method for saving detailed classification result.
"""

from ltsvm.twinsvm import TSVM, MCTSVM, OVO_TSVM
from ltsvm.misc import progress_bar_gs, time_fmt
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#from itertools import product
from datetime import datetime
import numpy as np
import pandas as pd
import os
# import time


def eval_metrics(y_true, y_pred):

    """
    It computes common evaluation metrics such as Accuracy, Recall, Precision,
    F1-measure, and elements of the confusion matrix.
    
    Parameters
    ----------
    y_true : array-like 
        Target values of samples.
        
    y_pred : array-like 
        Predicted class lables.
        
    Returns
    -------
    tp : int
        True positive.
        
    tn : int
        True negative.
        
    fp : int
        False positive.
        
    fn : int
        False negative.
        
    accuracy : float
        Overall accuracy of the model.
        
    recall_p : float
        Recall of positive class.
        
    precision_p : float
        Precision of positive class.
        
    f1_p : float
        F1-measure of positive class.
        
    recall_n : float
        Recall of negative class.
        
    precision_n : float
        Precision of negative class.
        
    f1_n : float
        F1-measure of negative class.
    """        
    
    # Elements of confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(y_true.shape[0]):
        
        # True positive 
        if y_true[i] == 1 and y_pred[i] == 1:
            
            tp = tp + 1
        
        # True negative 
        elif y_true[i] == -1 and y_pred[i] == -1:
            
            tn = tn + 1
        
        # False positive
        elif y_true[i] == -1 and y_pred[i] == 1:
            
            fp = fp + 1
        
        # False negative
        elif y_true[i] == 1 and y_pred[i] == -1:
            
            fn = fn + 1
            
    # Compute total positives and negatives
    positives = tp + fp
    negatives = tn + fn

    # Initialize
    accuracy = 0
    # Positive class
    recall_p = 0
    precision_p = 0
    f1_p = 0
    # Negative class
    recall_n = 0
    precision_n = 0
    f1_n = 0
    
    try:
        
        accuracy = (tp + tn) / (positives + negatives)
        # Positive class
        recall_p = tp / (tp + fn)
        precision_p = tp / (tp + fp)
        f1_p = (2 * recall_p * precision_p) / (precision_p + recall_p)
        
        # Negative class
        recall_n = tn / (tn + fp)
        precision_n = tn / (tn + fn)
        f1_n = (2 * recall_n * precision_n) / (precision_n + recall_n)
        
    except ZeroDivisionError:
        
        pass # Continue if division by zero occured


    return tp, tn, fp, fn, accuracy * 100 , recall_p * 100, precision_p * 100, f1_p * 100, \
           recall_n * 100, precision_n * 100, f1_n * 100


class Validator:

    """
    It evaluates the TwinSVM model based on the specified evaluation method.
    
    Parameters
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training feature vectors, where n_samples is the number of samples
        and n_features is the number of features.
        
    y_train : array-like, shape (n_samples,)
        Target values or class labels.
        
    validator_type : tuple
        A two-element tuple which contains type of evaluation method and its
        parameter. Example: ('CV', 5) -> 5-fold cross-validation,
        ('t_t_split', 30) -> 30% of samples for test set.
        
    obj_tsvm : object
        A TwinSVM model. It can be an instace of :class:`TSVM <twinsvm.TSVM>`,
        :class:`MCTSVM <twinsvm.MCTSVM>`, or :class:`OVO_TSVM <twinsvm.OVO_TSVM>`.
    """

    def __init__(self, X_train, y_train, validator_type, obj_tsvm):

        self.train_data = X_train
        self.labels_data = y_train
        self.validator = validator_type
        self.obj_TSVM = obj_tsvm

    def cv_validator(self, dict_param):

        """
        It evaluates the TwinSVM model using the cross-validation method.
        
        Parameters
        ----------
        dict_param : dict 
            Values of hyper-parameters for the TwinSVM model.
            
        Returns
        -------
        float
            Mean accuracy of the model.
            
        float
            Standard deviation of accuracy.
            
        dict
            Evaluation metrics such as Recall, Percision and F1-measure for both
            classes as well as elements of the confusion matrix.
        """
        
        # Set parameters of TSVM classifer
        self.obj_TSVM.set_params(**dict_param)
        
        # K-Fold Cross validation, divide data into K subsets
        k_fold = KFold(self.validator[1])    

        # Store result after each run
        mean_accuracy = []
        # Postive class
        mean_recall_p, mean_precision_p, mean_f1_p = [], [], []
        # Negative class
        mean_recall_n, mean_precision_n, mean_f1_n = [], [], []
        
        # Count elements of confusion matrix
        tp, tn, fp, fn = 0, 0, 0, 0
        
        # Train and test TSVM classifier K times
        for train_index, test_index in k_fold.split(self.train_data):

            # Extract data based on index created by k_fold
            X_train = np.take(self.train_data, train_index, axis=0) 
            X_test = np.take(self.train_data, test_index, axis=0)

            y_train = np.take(self.labels_data, train_index, axis=0)
            y_test = np.take(self.labels_data, test_index, axis=0)

            # fit - create two non-parallel hyperplanes
            self.obj_TSVM.fit(X_train, y_train)

            # Predict
            output = self.obj_TSVM.predict(X_test)

            accuracy_test = eval_metrics(y_test, output)

            mean_accuracy.append(accuracy_test[4])
            # Positive cass
            mean_recall_p.append(accuracy_test[5])
            mean_precision_p.append(accuracy_test[6])
            mean_f1_p.append(accuracy_test[7])
            # Negative class    
            mean_recall_n.append(accuracy_test[8])
            mean_precision_n.append(accuracy_test[9])
            mean_f1_n.append(accuracy_test[10])

            # Count
            tp = tp + accuracy_test[0]
            tn = tn + accuracy_test[1]
            fp = fp + accuracy_test[2]
            fn = fn + accuracy_test[3]

        return np.mean(mean_accuracy), np.std(mean_accuracy), {**{'accuracy': np.mean(mean_accuracy),
                      'acc_std': np.std(mean_accuracy),'recall_p': np.mean(mean_recall_p),
                      'r_p_std': np.std(mean_recall_p), 'precision_p': np.mean(mean_precision_p),
                      'p_p_std': np.std(mean_precision_p), 'f1_p': np.mean(mean_f1_p),
                      'f1_p_std': np.std(mean_f1_p), 'recall_n': np.mean(mean_recall_n),
                      'r_n_std': np.std(mean_recall_n), 'precision_n': np.mean(mean_precision_n),
                      'p_n_std': np.std(mean_precision_n), 'f1_n': np.mean(mean_f1_n),
                      'f1_n_std': np.std(mean_f1_n), 'tp': tp, 'tn': tn, 'fp': fp,
                      'fn': fn}, **dict_param}

    def split_tt_validator(self, dict_param):
        
        """
        It evaluates the TwinSVM model using the train/test split method.
        
        Parameters
        ----------
        dict_param : dict
            Values of hyper-parameters for the TwinSVM model.
            
        Returns
        -------
        float
            Accuracy of the model.
            
        float
            Zero standard deviation.
            
        dict
            Evaluation metrics such as Recall, Percision and F1-measure for both
            classes as well as elements of the confusion matrix.
        """

        # Set parameters of TSVM classifer
        self.obj_TSVM.set_params(**dict_param)

        X_train, X_test, y_train, y_test = train_test_split(self.train_data, \
                                           self.labels_data, test_size=self.validator[1], \
                                           random_state=42)

        # fit - create two non-parallel hyperplanes
        self.obj_TSVM.fit(X_train, y_train)

        output = self.obj_TSVM.predict(X_test)

        tp, tn, fp, fn, accuracy, recall_p, precision_p, f1_p, recall_n, precision_n, \
        f1_n = eval_metrics(y_test, output)

       # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c1=5, c2=6, gamma=7,
       # m_r_n=8, m_p_n=9, m_f1_n=10, tp=11, tn=12, fp=13, fn=14,
        return accuracy, 0.0, {**{'accuracy': accuracy, 'recall_p': recall_p,
               'precision_p': precision_p, 'f1_p': f1_p, 'recall_n': recall_n,
               'precision_n': precision_n, 'f1_n': f1_n, 'tp': tp, 'tn': tn,
               'fp': fp, 'fn': fn}, **dict_param}

    def cv_validator_mc(self, dict_param):

        """
        It evaluates the multi-class TwinSVM model using the cross-validation.
        
        Parameters
        ----------
        dict_param : dict 
            Values of hyper-parameters for the multiclss TwinSVM model.
            
        Returns
        -------
        float
            Accuracy of the model.
            
        float
            Zero standard deviation.
            
        dict
            Evaluation metrics such as Recall, Percision and F1-measure.
        """

        # Set parameters of multiclass TSVM classifer
        self.obj_TSVM.set_params(**dict_param)

        # K-Fold Cross validation, divide data into K subsets
        k_fold = KFold(self.validator[1])    

        # Store result after each run
        mean_accuracy = []
        
        # Evaluation metrics
        mean_recall, mean_precision, mean_f1 = [], [], []
        
        # Train and test multiclass TSVM classifier K times
        for train_index, test_index in k_fold.split(self.train_data):

            # Extract data based on index created by k_fold
            X_train = np.take(self.train_data, train_index, axis=0) 
            X_test = np.take(self.train_data, test_index, axis=0)

            y_train = np.take(self.labels_data, train_index, axis=0)
            y_test = np.take(self.labels_data, test_index, axis=0)

            # fit - creates K-binary TSVM classifier
            self.obj_TSVM.fit(X_train, y_train)

            # Predict
            output = self.obj_TSVM.predict(X_test)

            mean_accuracy.append(accuracy_score(y_test, output) * 100)
            mean_recall.append(recall_score(y_test, output, average='micro') * 100)
            mean_precision.append(precision_score(y_test, output, average='micro') * 100)
            mean_f1.append(f1_score(y_test, output, average='micro') * 100)

        return np.mean(mean_accuracy), np.std(mean_accuracy), {**{'accuracy':
               np.mean(mean_accuracy), 'acc_std': np.std(mean_accuracy),
               'micro_recall': np.mean(mean_recall), 'm_rec_std': np.std(mean_recall),
               'micro_precision': np.mean(mean_precision), 'm_prec_std':
               np.std(mean_precision), 'mirco_f1': np.mean(mean_f1), 'm_f1_std':
               np.std(mean_f1)}, **dict_param}

    def choose_validator(self):

        """
        It selects the appropriate evaluation method based on the input
        paramters.
        
        Returns
        -------
        object
            An evaluation method for assesing the model's performance.
        """

        if isinstance(self.obj_TSVM, TSVM):  # Binary TSVM

            if self.validator[0] == 'CV':

                return self.cv_validator

            elif self.validator[0] == 't_t_split':

                return self.split_tt_validator

        else:  # Multi-class TSVM

            if self.validator[0] == 'CV':

                return self.cv_validator_mc


def search_space(kernel_type, class_type, c_l_bound, c_u_bound, rbf_lbound, \
                 rbf_ubound, step=1):

    """
    It generates all combination of search elements based on the given range of 
    hyperparameters.
    
    Parameters
    ----------
    kernel_type : str, {'linear', 'RBF'}
        Type of the kernel function which is either 'linear' or 'RBF'.
        
    class_type : str, {'binary', 'ovo', 'ova'}
        Type of classification.
    
    c_l_bound : int
        Lower bound for C penalty parameter.
    
    c_u_bound : int
        Upper bound for C penalty parameter.
        
    rbf_lbound : int
        Lower bound for gamma parameter which is the hyperparameter of the RBF
        kernel function.
          
    rbf_ubound : int
        Upper bound for gamma parameter.
    
    step : int, optinal (default=1)
        Step size to increase power of 2. 
    
    Returns
    -------
    list
        Search elements.
        
    Examples
    --------
    >>> from ltsvm import eval_classifier
    >>> eval_classifier.search_space('RBF', 'binary', -1, 1, -1, 1)
    [{'C1': 0.5, 'C2': 0.5, 'gamma': 0.5}...
    {'C1': 1.0, 'C2': 1.0, 'gamma': 0.5}... {'C1': 2.0, 'C2': 2.0, 'gamma': 2.0}]
    """

    c_range = [2**i for i in np.arange(c_l_bound, c_u_bound+1, step,
                                         dtype=np.float)]
    
    gamma_range = [2**i for i in np.arange(rbf_lbound, rbf_ubound+1, step,
                   dtype=np.float)] if kernel_type == 'RBF' else [1]
    
    if class_type == 'binary' or class_type == 'ovo':
        
        param_grid = ParameterGrid({'C1': c_range, 'C2': c_range,
                                    'gamma': gamma_range})

    elif class_type == 'ova':
        
        param_grid = ParameterGrid({'C': c_range, 'gamma': gamma_range})

    return list(param_grid)


def grid_search(search_space, func_validator):

    """
    It does grid search to find the optimcal values of hyperparameters for the
    TwinSVM model, which results in the best classfication accuracy.
    
    Parameters
    ----------
    search_space : list
        All combination of search elements.
    
    func_validator : object
        An evaluation method for assesing the TwinSVM model's performance.
            
    Returns
    -------
    list
        Classification results of the TwinSVM classifier using different set of
        hyperparameters.
    """

    # Store 
    result_list = []
    
    # Max accuracy
    max_acc, max_acc_std = 0, 0

    # Total number of search elements
    search_total = len(search_space)

	# Dispaly headers and progress bar
#    print("TSVM-%s    Dataset: %s    Total Search Elements: %d" % (kernel_type, \
#          file_name, search_total))
    progress_bar_gs(0, search_total, '0:00:00', (0.0, 0.0), (0.0, 0.0), prefix='', \
                    suffix='')

    start_time = datetime.now()

    run = 1   

    # Ehaustive Grid search for finding optimal parameters
    for element in search_space:

        try:

            #start_time = time.time()

            # Save result after each run
            acc, acc_std, result = func_validator(element)

            #end = time.time()
            
            # For debugging purpose
            #print('Acc: %.2f+-%.2f | params: %s' % (acc, acc_std, str(result)))

            result_list.append(result)

            # Save best accuracy
            if acc > max_acc:
                
                max_acc = acc
                max_acc_std = acc_std       
            
            elapsed_time = datetime.now() - start_time
            progress_bar_gs(run, search_total, time_fmt(elapsed_time.seconds), \
                            (acc, acc_std), (max_acc, max_acc_std), prefix='', suffix='') 

            run = run + 1

        # Some parameters cause errors such as Singular matrix        
        except np.linalg.LinAlgError:
        
            run = run + 1


    return result_list


def save_result(file_name, validator_obj, gs_result, output_path):

    """
    It saves the detailed classification results in a spreadsheet file (Excel).

    Parameters
    ----------
    file_name : str
        Name of the spreadsheet file.
        
    validator_obj : object
        The evaluation method that was used for the assesment of the TwinSVM
        classifier.
    
    gs_result : list 
        Classification results of the TwinSVM classifier using different set of
        hyperparameters.
        
    output_path : str
        Path at which the spreadsheet file will be saved.

    Returns
    -------
    str
        Path to the saved spreadsheet (Excel) file.
    """

    column_names = {'binary': {'CV': ['accuracy', 'acc_std', 'recall_p', 'r_p_std', 'precision_p', 'p_p_std', \
                           'f1_p', 'f1_p_std', 'recall_n', 'r_n_std', 'precision_n', 'p_n_std', 'f1_n',\
                           'f1_n_std', 'tp', 'tn', 'fp', 'fn'],  #'c1', 'c2','gamma'],
                    't_t_split': ['accuracy', 'recall_p', 'precision_p', 'f1_p', 'recall_n', 'precision_n', \
                                  'f1_n', 'tp', 'tn', 'fp', 'fn']},  #, 'c1', 'c2','gamma']},
                    'multiclass':{'CV': ['accuracy', 'acc_std', 'micro_recall', 'm_rec_std', 'micro_precision', \
                                         'm_prec_std', 'mirco_f1', 'm_f1_std']}} #'C', 'gamma']

    # (Name of validator, validator's attribute) - ('CV', 5-folds)
    validator_type, validator_attr = validator_obj.validator              

    output_file = os.path.join(output_path, "%s_%s_%s_%s_%s.xlsx") % (validator_obj.obj_TSVM.cls_name,
                              validator_obj.obj_TSVM.kernel, "%d-F-CV" %
                              validator_attr if validator_type == 'CV' else 'Tr%d-Te%d' % \
                  ((1.0 - validator_attr) * 100, validator_attr * 100),
                  file_name, datetime.now().strftime('%Y-%m-%d %H-%M'))

    excel_file = pd.ExcelWriter(output_file, engine='xlsxwriter')
    
    # columns=column_names['binary' if \isinstance(validator_obj.obj_TSVM, TSVM) else 'multiclass'][validator_type]
    result_frame = pd.DataFrame(gs_result, columns=column_names['binary' if 
                                isinstance(validator_obj.obj_TSVM, TSVM) else
                                'multiclass'][validator_type] + validator_obj.obj_TSVM.get_params_names()) 

    result_frame.to_excel(excel_file, sheet_name='Sheet1', index=False)

    excel_file.save()

    return os.path.abspath(output_file)  


def initializer(user_input_obj):

    """
    It passes a user's input to the functions and classes for solving a
    classification task. The steps that this function performs can be summarized
    as follows:
        
    #. Specifies a TwinSVM classifier based on the user's input.
    #. Chooses an evaluation method for assessment of the classifier.
    #. Computes all the combination of search elements.
    #. Computes the evaluation metrics for all the search element using grid search.
    #. Saves the detailed classification results in a spreadsheet file (Excel).
    
    Parameters
    ----------
    user_input_obj : object 
        An instance of :class:`UserInput` class which holds the user input.
    """

    if user_input_obj.class_type == 'binary':

        tsvm_obj = TSVM(user_input_obj.kernel_type, user_input_obj.rect_kernel)

    elif user_input_obj.class_type == 'ovo':

        tsvm_obj = OVO_TSVM(user_input_obj.kernel_type)
        
    elif user_input_obj.class_type == 'ova':
        
        tsvm_obj = MCTSVM(user_input_obj.kernel_type)

    validate = Validator(user_input_obj.X_train, user_input_obj.y_train, \
                         user_input_obj.test_method_tuple, tsvm_obj)

    search_elements = search_space(user_input_obj.kernel_type, user_input_obj.class_type, \
                      user_input_obj.lower_b_c, user_input_obj.upper_b_c, user_input_obj.lower_b_u, \
                      user_input_obj.upper_b_u)

    # Display headers
    print("%s-%s    Dataset: %s    Total Search Elements: %d" % (tsvm_obj.cls_name,
          user_input_obj.kernel_type, user_input_obj.filename, len(search_elements)))

    result = grid_search(search_elements, validate.choose_validator())

    try:

        return save_result(user_input_obj.filename, validate, result, user_input_obj.result_path)

    except FileNotFoundError:

        os.makedirs('result')

        return save_result(user_input_obj.filename, validate, result, user_input_obj.result_path)
