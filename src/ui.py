#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.1.0-alpha - 2018-05-24
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: ui.py
User interface of LightTwinSVM program is implemented in this module.

"""


from eval_classifier import grid_search
from dataproc import read_data, read_libsvm
from tkinter import Tk, filedialog
from os import path
import time
import numpy as np


def get_check_range(message):
    
    """
        This function gets the user input and checks it for grid search's parameter
        
        Input:
            message: For printing.
    
    """
    
    while True:
    
        try:
            
            print(message)
        
            c_range = input("-> ").split(' ')
                    
            if len(c_range) == 2:
                
                c_l, c_u = int(c_range[0]), int(c_range[1])
                    
                if c_l < c_u:
                        
                    return c_l, c_u
            
            print("Wrong input! Try again.")
            
            continue
        
        except ValueError:
            
            print("Wrong input! Try again.")
            
            continue


def program_ui():

    """
    User interface of the program is implemented in this function.
    It processes user input.
    
    """
    
    # Training samples and its corresponding category
    X_train, y_train = None, None
    
    global kernel_type
    global test_method_tuple
    global k_fold
    global lower_b_c, upper_b_c
    global lower_b_u, upper_b_u
    lower_b_u, upper_b_u = None, None
    
    # Printing general info on program    
    print("""LightTwinSVM Program - Simple and Fast
Version: 0.1.0-alpha - 2018-05-24
Developer: Mir, A. (mir-am@hotmail.com)
Paper's authors: Khemchandani, R. & Chandra, S.
License: GNU General Public License v3.0
*************************************************\n""")
    
    input("Press Enter to start the program...\n")
    
    while True:
        
        print("Step 1/4: Please select your dataset... (CSV and LIBSVM file supported.)")
        
        # Step 1: User selects his/her dataset
        while True:
            
            input("Press Enter to open dataset...")
            
            # Opens a simple dialog for choosing dataset
            root = Tk()
            root.withdraw()
            dataset_path = filedialog.askopenfilename(title="Choose your dataset", \
                           filetypes=(('CSV file', '*.csv'), ('LIBSVM data file', '*.libsvm'),))
            
            if dataset_path != () and path.isfile(dataset_path):
            
                file_name, file_ext = path.splitext(dataset_path)
                
                start_t = time.time()

                if file_ext == '.csv':
                
		           # Reading and processing user dataset
                    try:
						     
                        header = False
						
                        # First assume that dataset has no header names.
                        X_train, y_train, file_name = read_data(dataset_path, header)

                    except ValueError:

                        print("Lookslike your dataset has header names.")

                        header = True
                        start_t = time.time()
						
                        X_train, y_train, file_name = read_data(dataset_path, header)
                
                elif file_ext == '.libsvm':

                    X_train, y_train, file_name = read_libsvm(dataset_path)
        
                
                print("Your dataset \"%s\" is successfully loaded in %.2f seconds..." % \
                     (file_name, time.time() - start_t))
                
                print("No. of samples: %d|No. of features: %d|No. of classes: %d|Datatype: %s\n" % \
                     (X_train.shape[0], X_train.shape[1], np.unique(y_train).size, \
                     'CSV' if file_ext == '.csv' else 'LIBSVM'))
            
            else:
            
                print("Error: Invalid address. Try again.")
        
                continue
            
            
            break
        
        # Step 2: Select a kernel function type
        while True:
                       
            print("Step 2/4: Choose a kernel function:(Just type the number. e.g 1)\n1-Linear\n2-RBF")
            
            kernel_choice = input("-> ")
            
            if '1' in kernel_choice:
                
                kernel_type = 'linear'
            
            elif '2' in kernel_choice:
                
                kernel_type = 'RBF'
                
            else:
                
                print("Wrong input! Try again.")
                
                continue
            
            break
        
        # Step 3: Testing methodolgy    
        while True:
            
            print("Step 3/4: Choose a test methodolgy:(Just type the number. e.g 1)\n1-K-fold cross validation\n2-Train/test split")
            
            test_choice = input("-> ")
            
            try:
            
                if '1' in test_choice:
                    
                    test_method = 'CV'
                    
                    print("Determine number of folds for cross validaton: (e.g. 5)")
                    
                        
                    k_fold = int(input("-> "))
                
                    if not(k_fold >= 2 and k_fold <= 10):
                        
                        print("Number of folds should be 2 <= k <= 10. Try again.\n")
                        
                        continue
                    
                    test_method_tuple = (test_method, k_fold)
                 
                
                elif '2' in test_choice:
                    
                    test_method = 't_t_split'
                    
                    # Support for split test/train soon!
                    
                    print("Determine percentage of training set: (e.g. 70)")
                    
                    train_set_percent = int(input("-> "))
                    
                    if not(train_set_percent >= 1 and train_set_percent <= 99):
                        
                        print("Percentage of training set should be between 1 and 99 percent. Try again.\n")
                        
                        continue
                    
                    test_method_tuple = (test_method, 100 - train_set_percent)
                    
                else:
                    
                    print("Wrong input! Try again.")
                    
                    continue
                
            except ValueError:
                
                print("Wrong input! Try again.")
                        
                continue
            
            break
        
        # Step 4: User types the range of paramters for grid search               
        # Lower and upper bound of C paramter
        lower_b_c, upper_b_c = get_check_range("Step 4/4:Type the range of C penalty parameter for grid search:\n(Two integer numbers separated by space. e.g. -> -5 5")
                       
        if kernel_type == 'RBF':
            
            # Lower and upper bound of gamma paramter
            lower_b_u, upper_b_u = get_check_range("Type the range of gamma parameter for RBF function.\n(Two integer numbers separated by space. e.g. -> -10 2")
                    
        print("Do you confirm the following settings for running TwinSVM classifier?(y/n)")   
        print("Dataset: %s\nKernel function: %s\nTest method: %s\nRange of parameters for grid search:\nC: 2^%d to 2^%d%s "  \
              % (file_name, kernel_type, "%d-Fold cross validation" % k_fold if test_method == 'CV' else "Train/test split (%d%%/%d%%)" % \
                 (train_set_percent, 100 - train_set_percent), lower_b_c, upper_b_c, " | Gamma: 2^%d to 2^%d" % (lower_b_u, upper_b_u) if kernel_type == 'RBF' else ''))
        
        if 'y' in input("-> "):
            
            start_time = time.time()
            
            result_file = grid_search(test_method_tuple, kernel_type, X_train, y_train, lower_b_c, upper_b_c, \
                        lower_b_u, upper_b_u, 1, file_name)
            
            print("Search finished in %.2f Sec." % (time.time() - start_time))
            
            print("The spreadsheet file containing the result is in:\n%s\n" % result_file)
            
            print("Do you want to run TwinSVM classifier again? (y/n)")
            
            if 'y' in input("-> "):
                
                continue
            
            else:
                
                break
        
        else:
            
            continue
        