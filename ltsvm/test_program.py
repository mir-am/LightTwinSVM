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

from eval_classifier import Validator, initializer
from ui import UserInput
from dataproc import read_data, read_libsvm
from twinsvm import TSVM, MCTSVM, OVO_TSVM
import unittest


def print_test_info(test_input):

    """
    It runs the test and prints info about it.
    """

    test_info = (test_input.filename, test_input.kernel_type if test_input.rect_kernel == 1 else \
                 'Rectangular kernel(Using %d %% of samples)' % (test_input.rect_kernel * 100), '%d-Fold-CV' % \
                 test_input.test_method_tuple[1] if test_input.test_method_tuple[0] == 'CV' else \
                 'Tr%d' % (test_input.test_method_tuple[1]), test_input.lower_b_c, \
                 test_input.upper_b_c, ',u:2^%d-2^%d' % (test_input.lower_b_u, \
                 test_input.upper_b_u) if test_input.kernel_type == 'RBF' else '')

    print("Data: %s|Kernel: %s|Test: %s|C:2^%d-2^%d%s" % test_info)

    initializer(test_input)

    print("*******************************************************")


class TestProgram(unittest.TestCase):

    """
        It checks different functionalities of the program.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.input = UserInput()
        # Default settings for unit test - Binary Classification
        # Dataset
        self.input.X_train, self.input.y_train, self.input.filename = \
        read_data("./dataset/pima-indian.csv")
        # Lower and upper bounds of parameters
        self.input.lower_b_c, self.input.upper_b_c = -2, 2
        self.input.lower_b_u, self.input.upper_b_u = -2, 2
        self.input.filename = 'UnitTest_' + self.input.filename
        self.input.class_type = 'binary'
        self.input.result_path = './result'

        self.input_mc = UserInput()
        # Default settings for unit test - multiclass Classification
        # Dataset
        self.input_mc.X_train, self.input_mc.y_train, self.input_mc.filename = \
        read_data('./dataset/wine.csv')
        # Lower and upper bounds of parameters
        self.input_mc.lower_b_c, self.input_mc.upper_b_c = -2, 2
        self.input_mc.lower_b_u, self.input_mc.upper_b_u = -2, 2
        self.input_mc.filename = 'UnitTest_' + self.input_mc.filename
        self.input_mc.class_type = 'multiclass'
        self.input.result_path = './result'

        self.k_folds = 5
        self.train_set_size = 90


    def test_train_TSVM_linear(self):

        """
        It trains TSVM classifier with Linear kernel
        """

        # Default arguments
        tsvm_classifier = TSVM()
        tsvm_classifier.fit(self.input.X_train, self.input.y_train)
        tsvm_classifier.predict(self.input.X_train)

    def test_train_TSVM_RBF(self):

        """
        It trains TSVM classifier with RBF kernel
        """

        # Default arguments
        tsvm_classifier = TSVM(kernel_type='RBF')
        tsvm_classifier.fit(self.input.X_train, self.input.y_train)
        tsvm_classifier.predict(self.input.X_train)

    def test_linear_Validator_CV(self):

        """
        It applies cross validation on Linear TSVM
        """

        tsvm_classifier = TSVM()
        validate = Validator(self.input.X_train, self.input.y_train, ('CV', \
                             self.k_folds), tsvm_classifier)

        func = validate.choose_validator()
        func()

    def test_RBF_Validator_CV(self):

        """
        It applies cross validation on non-Linear TSVM
        """

        tsvm_classifier = TSVM(kernel_type='RBF')
        validate = Validator(self.input.X_train, self.input.y_train, ('CV', \
                             self.k_folds), tsvm_classifier)

        func = validate.choose_validator()
        func()

    def test_linear_Validator_ttsplit(self):

        """
        It applies train/test split on Linear TSVM
        """

        tsvm_classifier = TSVM(kernel_type='linear')
        validate = Validator(self.input.X_train, self.input.y_train, ('t_t_split', \
                             self.train_set_size), tsvm_classifier)

        func = validate.choose_validator()
        func()

    def test_RBF_Validator_ttsplit(self):

        """
        It applies train/test split in non-linear TSVM
        """
        tsvm_classifier = TSVM(kernel_type='RBF')
        validate = Validator(self.input.X_train, self.input.y_train, ('t_t_split', \
                             self.train_set_size), tsvm_classifier)

        func = validate.choose_validator()
        func()

    def test_linear_CV_gridsearch(self):

        """
            It checks Linear kernel, CrossValidation and grid search.
        """

        self.input.kernel_type = 'linear'
        self.input.test_method_tuple = ('CV', self.k_folds)

        print_test_info(self.input)

    def test_linear_ttsplit_gridsearch(self):

        """
            It checks Linear kernel, Train/Test split and grid search.
        """

        self.input.kernel_type = 'linear'
        self.input.test_method_tuple = ('t_t_split', self.train_set_size)

        print_test_info(self.input)

    def test_RBF_CV_gridsearch(self):

        """
            It checks RBF kernel, CrossValidation and grid search.
        """

        self.input.kernel_type = 'RBF'
        self.input.test_method_tuple = ('CV', self.k_folds)

        print_test_info(self.input)

    def test_RBF_ttsplit_gridsearch(self):

        """
            It checks RBF kernel, Train/Test split and grid search.
        """

        self.input.kernel_type = 'RBF'
        self.input.test_method_tuple = ('t_t_split', self.train_set_size)

        print_test_info(self.input)

    def test_LIBSVM_linear_CV_girdsearch(self):

        """
        It checks linear kenel, CrossValidation and grid search with LIBSVM data
        """

        self.input.kernel_type = 'linear'
        self.input.test_method_tuple = ('CV', self.k_folds)

        # Keep dataset for other tests.
        temp = (self.input.X_train, self.input.y_train, self.input.filename)

        self.input.X_train, self.input.y_train, self.input.filename = read_libsvm("./dataset/heart.libsvm")

        print_test_info(self.input)

        self.input.X_train, self.input.y_train, self.input.filename = temp
        
    def test_rectangular_CV_gridsearch(self):

        """
        It checks rectangular kernel, CrossValidation and grid search
        """

        self.input.kernel_type = 'RBF'
        self.input.rect_kernel = 0.5  # Using 50% of samples for Rectangular kernel
        self.input.test_method_tuple = ('CV', self.k_folds)

        print_test_info(self.input)

        self.input.rect_kernel = 1  # Default value for RBF kernel

    def test_linear_MCTSVM(self):

        """
        It checks linear OVA TwinSVM
        """

        mctsvm_obj = MCTSVM()
        mctsvm_obj.fit(self.input_mc.X_train, self.input_mc.y_train)
        mctsvm_obj.predict(self.input_mc.X_train)

    #@unittest.skip('Singular Matrix!')
    def test_RBF_MCTSVM(self):

        """
        It checks non-linear OVA TwinSVM
        """

        mctsvm_obj = MCTSVM('RBF')
        mctsvm_obj.fit(self.input_mc.X_train, self.input_mc.y_train)
        mctsvm_obj.predict(self.input_mc.X_train)

    def test_linear_CV_gridsearch_MCTSVM(self):

        """
        It checks linear kernel, Crossvalidation and grid search with MCTSVM
        """

        self.input_mc.kernel_type = 'linear'
        self.input_mc.test_method_tuple = ('CV', self.k_folds)

        print_test_info(self.input_mc)

    def test_RBF_CV_gridsearch_MCTSVM(self):

        """
        It checks RBF kernel, Crossvalidation and grid search with MCTSVM
        """

        self.input_mc.kernel_type = 'RBF'
        self.input_mc.test_method_tuple = ('CV', self.k_folds)

        print_test_info(self.input_mc)
        
    def test_linear_OVO_TSVM(self):
        
        """
        It checks linear OVO TwinSVM
        """
        
        ovo_tsvm_obj = OVO_TSVM()
        ovo_tsvm_obj.fit(self.input_mc.X_train, self.input_mc.y_train)
        ovo_tsvm_obj.predict(self.input_mc.X_train)
        
    def test_RBF_OVO_TSVM(self):
        
        """
        It checks non-linear OVO TwinSVM
        """
        
        ovo_tsvm_obj = OVO_TSVM('RBF')
        ovo_tsvm_obj.fit(self.input_mc.X_train, self.input_mc.y_train)
        ovo_tsvm_obj.predict(self.input_mc.X_train)


if __name__ == '__main__':

    unittest.main()
