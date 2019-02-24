#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LightTwinSVM Program - Simple and Fast
# Version: 0.5.0 - 2019-01-01
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

"""
In this module, methods are defined for evluating TwinSVM perfomance such as cross validation
train/test split, grid search and generating the detailed result.
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
        Input:
            
            y_true: True label of samples
            y_pred: Prediction of classifier for test samples
    
        output: Elements of confusion matrix and Evalaution metrics such as
        accuracy, precision, recall and F1 score
    
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
    It applies a test method such as cross validation on a classifier like TSVM
    """

    def __init__(self, X_train, y_train, validator_type, obj_tsvm):

        """
        It constructs and returns a validator 
        Input:
            X_train: Samples in dataset (2-d NumPy array)
            y_train: Labels of samples in dataset (1-d NumPy array)
            validator_type: Type of test methodology and its parameter.(Tuple - ('CV', 5))
            obj_tsvm:  Instance of TSVM classifier. (TSVM class)

        """

        self.train_data = X_train
        self.labels_data = y_train
        self.validator = validator_type
        self.obj_TSVM = obj_tsvm

    def cv_validator(self, dict_param):

        """
        It applies cross validation on instance of Binary TSVM classifier
        Input:
            dict_param: A dictionary of hyper-parameters (dict)
            
        output:
            Evaluation metrics such as accuracy, precision, recall and F1 score
            for each class.
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
        It trains TwinSVM classifier on random training set and tests the classifier
        on test set.
        output:
            Evaluation metrics such as accuracy, precision, recall and F1 score
            for each class.
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
        It applies cross validation on instance of multiclass TSVM classifier
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
        It returns choosen validator method.
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
    It generates combination of search elements for grid search
    Input:
        kernel_type: kernel function which is either linear or RBF
        c_l_bound, c_u_bound: Range of C penalty parameter for grid search(e.g 2^-5 to 2^+5)
        rbf_lbound, rbf_ubound: Range of gamma parameter
    Output:
        return search elements for grid search (List)
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
        It applies grid search which finds C and gamma paramters for obtaining
        best classification accuracy.
    
        Input:
           search_space: search_elements (List)
           func_validator: Validator function
            
        output:
            returns classification result (List)
    
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
        It saves detailed result in spreadsheet file(Excel).

        Input:
            file_name: Name of spreadsheet file
            col_names: Column names for spreadsheet file
            gs_result = result produced by grid search
            output_path: Path to store the spreadsheet file.

        output:
            returns path of spreadsheet file

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
    It gets user input and passes function and classes arguments to run the program
    Input:
        user_input_obj: User input (UserInput class)
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

    # Dispaly headers
    print("%s-%s    Dataset: %s    Total Search Elements: %d" % (tsvm_obj.cls_name,
          user_input_obj.kernel_type, user_input_obj.filename, len(search_elements)))

    result = grid_search(search_elements, validate.choose_validator())

    try:

        return save_result(user_input_obj.filename, validate, result, user_input_obj.result_path)

    except FileNotFoundError:

        os.makedirs('result')

        return save_result(user_input_obj.filename, validate, result, user_input_obj.result_path)
