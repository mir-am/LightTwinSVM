#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: test_program.py
In this module, unit test is defined for checking the integrity of installation.

"""

from eval_classifier import grid_search
from dataproc import read_data, read_libsvm
import unittest


class TestProgram(unittest.TestCase):

    """
        It checks different functionalities of the program.    
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Default settings for unit test
        # Dataset
        self.training_data, self.labels, self.file_name = read_data("./dataset/pima-indian.csv")
        # Lower and upper bounds of parameters
        self.c_l_b, self.c_u_b = -2, 2
        self.rbf_l_b, self.rbf_u_b = -2, 2
        self.k_folds = 5
        self.train_set_size = 90
        self.output_file = 'UnitTest'
        self.result_path = './result'
        
    def test_linear_CV_gridsearch(self):
        
        """
            It checks Linear kernel, CrossValidation and grid search.
        """
        
        grid_search(('CV', self.k_folds), 'linear', self.training_data, self.labels, \
                    self.c_l_b, self.c_u_b, self.rbf_l_b, self.rbf_u_b, 1, self.output_file, \
                    self.result_path)
        
    def test_linear_ttsplit_gridsearch(self):
        
        """
            It checks Linear kernel, Train/Test split and grid search.
        """
        
        grid_search(('t_t_split', self.train_set_size), 'linear', self.training_data, self.labels, \
                    self.c_l_b, self.c_u_b, self.rbf_l_b, self.rbf_u_b, 1, self.output_file, \
                    self.result_path)
        
    def test_RBF_CV_gridsearch(self):
        
        """
            It checks RBF kernel, CrossValidation and grid search.
        """
        
        grid_search(('CV', self.k_folds), 'RBF', self.training_data, self.labels, \
                    self.c_l_b, self.c_u_b, self.rbf_l_b, self.rbf_u_b, 1, self.output_file, \
                    self.result_path)

    def test_RBF_ttsplit_gridsearch(self):
        
        """
            It checks RBF kernel, Train/Test split and grid search.
        """
        
        grid_search(('t_t_split', self.train_set_size), 'RBF', self.training_data, self.labels, \
                    self.c_l_b, self.c_u_b, self.rbf_l_b, self.rbf_u_b, 1, self.output_file, \
                    self.result_path)

    def test_LIBSVM_linear_CV_girdsearch(self):
        
        """
        It checks linear kenel, CrossValidatiob and grid search with LIBSVM data
        """
        
        X_train, y_train, file_name = read_libsvm("./dataset/heart.libsvm")

        grid_search(('CV', self.k_folds), 'linear', X_train, y_train, self.c_l_b, \
                    self.c_u_b, self.rbf_l_b, self.rbf_u_b, 1, self.output_file, \
                    self.result_path)
   

if __name__ == '__main__':
    
    unittest.main()
