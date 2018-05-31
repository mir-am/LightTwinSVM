#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: eval_classifier.py
In this module, methods are defined for evluating TwinSVM perfomance such as cross validation
train/test split, grid search and generating the detailed result.
"""


from twinsvm import TSVM
from misc import progress_bar_gs, time_fmt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from itertools import product
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
           
           
def cv_validate(kernel_type, train_data, labels_data, k_fold, c1=2**0, c2=2**0, \
                gamma=2**0):
    
    """
        It applies cross validation on TwinSVM
        
        input:
            kernel_type: kernel function which is either linear or RBF
            train_data: Samples in dataset
            labels_data: Labels of samples in dataset
            k_fold: Number of folds in cross validation
            c1, c2: Penalty parameters
            gamma: Paramter of RBF kernel function
            
        output:
            Evaluation metrics such as accuracy, precision, recall and F1 score
            for each class.
            
    """
    
    # Instance of TwinSVM
    tsvm_classifier = TSVM(kernel_type, c1, c2, gamma)
    
    # K-Fold Cross validation, divide data into K subsets
    k_fold = KFold(k_fold)    
        
    # Store result after each run
    mean_accuracy = []
    # Postive class
    mean_recall_p, mean_precision_p, mean_f1_p = [], [], []
    # Negative class
    mean_recall_n, mean_precision_n, mean_f1_n = [], [], []
    
    # Count elements of confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    #k_time = 1
    
    # Train and test LSTSVM K times
    for train_index, test_index in k_fold.split(train_data):
        
        # Extract data based on index created by k_fold
        X_train = np.take(train_data, train_index, axis=0) 
        X_test = np.take(train_data, test_index, axis=0)
        
        y_train = np.take(labels_data, train_index, axis=0)
        y_test = np.take(labels_data, test_index, axis=0)
                                     
        # fit - create two non-parallel hyperplanes
        tsvm_classifier.fit(X_train, y_train)
        
        # Predict
        output = tsvm_classifier.predict(X_test)
        
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
        
        #print("K_fold %d finished..." % k_time)
        
        #k_time = k_time + 1
    
    # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c1=5, c2=6, gamma=7,
    # m_r_n=8, m_p_n=9, m_f1_n=10, tp=11, tn=12, fp=13, fn=14, iter=15    
    return np.mean(mean_accuracy), np.std(mean_accuracy), [np.mean(mean_accuracy), np.std(mean_accuracy), \
           np.mean(mean_recall_p), np.std(mean_recall_p), np.mean(mean_precision_p), np.std(mean_precision_p), np.mean(mean_f1_p), \
           np.std(mean_f1_p), np.mean(mean_recall_n), np.std(mean_recall_n), np.mean(mean_precision_n), np.std(mean_precision_n), \
           np.mean(mean_f1_n), np.std(mean_f1_n), tp, tn, fp, fn, c1, c2, gamma if kernel_type == 'RBF' else '']



def split_train_test(kernel_type, train_data, labels_data, test_percent, c1=2**0, \
                     c2=2**0, gamma=2**0):
    
    """
        It trains TwinSVM classifier on random training set and tests the classifier
        on test set.
        
        input:
            kernel_type: kernel function which is either linear or RBF
            X_train: Training samples
            X_test: Test samples
            y_train: Labels of training samples
            y_test: Labels of test samples
            c1, c2: Penalty parameters
            gamma: Paramter of RBF kernel function
            
    
    """
    
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels_data, \
                                       test_size=test_percent, random_state=42)
        
    tsvm_classifier = TSVM(kernel_type, c1, c2, gamma)
    
    # fit - create two non-parallel hyperplanes
    tsvm_classifier.fit(X_train, y_train)
        
    output = tsvm_classifier.predict(X_test)
    
    tp, tn, fp, fn, accuracy, recall_p, precision_p, f1_p, recall_n, precision_n, \
    f1_n = eval_metrics(y_test, output)
    
     
   # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c1=5, c2=6, gamma=7,
   # m_r_n=8, m_p_n=9, m_f1_n=10, tp=11, tn=12, fp=13, fn=14,   
    return accuracy, 0.0, [accuracy, recall_p, precision_p, f1_p, recall_n, precision_n, \
                           f1_n, tp, tn, fp, fn, c1, c2, gamma if kernel_type == 'RBF' else '']


def grid_search(test_method, kernel_type, train_data, labels, c_l_bound, c_u_bound, \
                rbf_lbound, rbf_ubound, step, file_name, output_path="./result"):
    
    """
        It applies grid search which finds C and gamma paramters for obtaining
        best classification accuracy.
    
        Input:
            test_method: Evaluate clasifier with cross validation or train/test split
            (test_type, k_fold or percent of test set)
            kernel_type: kernel function which is either linear or RBF
            train_data: Samples for training classifier
            labels: Class label of samples
            k: Number of k folds for cross validation
            c_l_bound, c_u_bound: Range of C penalty parameter for grid search(e.g 2^-5 to 2^+5)
            rbf_lbound, rbf_ubound: Range of gamma parameter
            
        output:
            Creates an Excel file which contains detailed classification result 
    
    """
                    
    test_function = {'CV': cv_validate, 't_t_split': split_train_test}
    
    # Store 
    result_list = []
    
    # Max accuracy
    max_acc, max_acc_std = 0, 0
        
    # Search space
    c_range = [2 ** i for i in np.arange(c_l_bound, c_u_bound + 1, step, dtype=np.float)]
    
    search_space = list(product(*[c_range, c_range, ])) if kernel_type == 'linear' else \
                   list(product(*[c_range, c_range, [2 ** i for i in np.arange(rbf_lbound, \
                    rbf_ubound + 1, step, dtype=np.float)]]))
    
    # Total number of search elements
    search_total = len(search_space)

	# Dispaly headers and progress bar
    print("TSVM-%s    Dataset: %s    Total Search Elements: %d" % (kernel_type, \
          file_name, search_total))
    progress_bar_gs(0, search_total, '0:00:00', (0.0, 0.0), (0.0, 0.0), prefix='', \
                    suffix='')

    start_time = datetime.now()
    
    run = 1   
    
    # Ehaustive or Grid search for finding best C1 & C2   
    for element in search_space:
            
        try:
                    
            #start_time = time.time()
                                      
            # Save result after each run
            #acc, acc_std, result = cv_validate(kernel_type, train_data, labels, k, *element)
            acc, acc_std, result = test_function[test_method[0]](kernel_type, train_data, \
                                                labels, test_method[1], *element)
            
            #end = time.time()
                       
            result_list.append(result)
            
            # Save best accuracy
            if acc > max_acc:
                
                max_acc = acc
                max_acc_std = acc_std       
            
            elapsed_time = datetime.now() - start_time
            progress_bar_gs(run, search_total, time_fmt(elapsed_time.seconds), \
                            (acc, acc_std), (max_acc, max_acc_std), prefix='', suffix='')
            #print("TSVM-%s|Run: %d|%d|Data:%s|C1:2^%d, C2:2^%d%s|B-Acc:%.2f+-%.2f|Acc: %.2f+-%.2f|Time: %.2f Sec." % \
            #     (kernel_type, run, search_total, file_name, np.log2(element[0]), np.log2(element[1]), ',u:2^%d' % \
            #       np.log2(element[2]) if kernel_type == 'RBF' else '', max_acc, max_acc_std, acc, acc_std, \
            #       end - start_time))  
            
            run = run + 1
    
        # Some parameters cause errors such as Singular matrix        
        except np.linalg.LinAlgError:
            
            run = run + 1
                
            #print("TSVM-%s|Run: %d|%d|Data:%s|C1:2^%d, C2:2^%d%s|B-Acc:%.2f+-%.2f|Linear Algebra Error!" % \
                  #(kernel_type, run, search_total, file_name, np.log2(element[0]), np.log2(element[1]), ',u:2^%d' % \
                   #np.log2(element[2]) if kernel_type == 'RBF' else '', max_acc, max_acc_std))
    
    output_file = os.path.join(output_path, "TSVM_%s_%s_%s_%s.xlsx") % (kernel_type, "%d-F-CV" % test_method[1] if test_method[0] == 'CV' \
                  else 'Tr%d-Te%d' % (100 - test_method[1], test_method[1]), file_name, \
                  datetime.now().strftime('%Y-%m-%d %H-%M'))    
        
    return save_result(output_file, test_method[0], result_list)


def save_result(file_name, col_names, gs_result):
    
    """
        It saves detailed result in spreadsheet file(Excel).
        
        Input:
            
            file_name: Name of spreadsheet file
            col_names: Column names for spreadsheet file
            gs_result = result produced by grid search
            
        output:
            
            returns path of spreadsheet file
    
    """
    
    column_names = {'CV': ['accuracy', 'acc_std', 'recall_p', 'r_p_std', 'precision_p', 'p_p_std', \
                           'f1_p', 'f1_p_std', 'recall_n', 'r_n_std', 'precision_n', 'p_n_std', 'f1_n',\
                           'f1_n_std', 'tp', 'tn', 'fp', 'fn', 'c1', 'c2','gamma'],
                    't_t_split': ['accuracy', 'recall_p', 'precision_p', 'f1_p', 'recall_n', 'precision_n', \
                                  'f1_n', 'tp', 'tn', 'fp', 'fn', 'c1', 'c2','gamma']}
                    
                    
    excel_file = pd.ExcelWriter(file_name, engine='xlsxwriter')
    
    # panda data frame
    result_frame = pd.DataFrame(gs_result, columns=column_names[col_names]) 

    result_frame.to_excel(excel_file, sheet_name='Sheet1', index=False)
    
    excel_file.save()
    
    return os.path.abspath(file_name)  
