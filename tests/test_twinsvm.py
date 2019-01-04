#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightTwinSVM Program - Simple and Fast
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

This test module tests the functionalities of twinsvm.py module
"""

# A temprory workaround to import LightTwinSVM for running tests
import sys
sys.path.append('./')

from ltsvm.twinsvm import TSVM, MCTSVM, OVO_TSVM
import unittest

class TestTwinSVM(unittest.TestCase):
    
    """
        It tests the estimators in twinsvm.py module
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def test_tsvm_set_get_params_linear(self):
        
        """
        It checks that set_params and get_params works correctly for TSVM-Linear
        """
        
        expected_output = {'gamma': 1, 'C1': 2, 'rect_kernel': 1, 'C2': 0.5,
                           'kernel': 'linear'}
        
        tsvm_cls = TSVM('linear')
        tsvm_cls.set_params(**{'C1': 2, 'C2':0.5})
        
        self.assertEqual(tsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_tsvm_set_get_params_rbf(self):
        
        """
        It checks that set_params and get_params works correctly for TSVM-RBF
        """
        
        expected_output = {'C2': 0.25, 'C1': 2, 'rect_kernel': 1,
                           'gamma': 0.125, 'kernel': 'RBF'}
        
        tsvm_cls = TSVM('RBF')
        tsvm_cls.set_params(**{'C1': 2, 'C2': 0.25, 'gamma': 0.125})
        
        self.assertEqual(tsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_mctsvm_set_get_params_linear(self):
        
        """
        It checks that set_params and get_params works correctly for MCTSVM-Linear
        """
        
        expected_output = {'kernel': 'linear', 'C': 0.5, 'gamma': 1}
        
        mctsvm_cls = MCTSVM('linear')
        mctsvm_cls.set_params(**{'C': 0.5})
        
        self.assertEqual(mctsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_mctsvm_set_get_params_rbf(self):
        
        """
        It checks that set_params and get_params works correctly for MCTSVM-RBF
        """
        
        expected_output = {'C': 2, 'gamma': 0.125, 'kernel': 'RBF'}
        
        mctsvm_cls = MCTSVM('RBF')
        mctsvm_cls.set_params(**{'C': 2, 'gamma': 0.125})
        
        self.assertEqual(mctsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_ovotsvm_set_get_params_linear(self):
        
        """
        It checks that set_params and get_params works correctly for OVO-TSVM-Linear
        """
        
        expected_output = {'C2': 2, 'gamma': 1, 'kernel': 'linear', 'C1': 0.5}
        
        ovotsvm_cls = OVO_TSVM('linear')
        ovotsvm_cls.set_params(**{'C1': 0.5, 'C2': 2})
        
        self.assertEqual(ovotsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_ovotsvm_set_get_params_rbf(self):
        
        """
        It checks that set_params and get_params works correctly for OVO-TSVM-RBF
        """
        
        expected_output = {'C2': 2, 'kernel': 'RBF', 'gamma': 0.125, 'C1': 0.5}
        
        ovotsvm_cls = OVO_TSVM('RBF')
        ovotsvm_cls.set_params(**{'C1': 0.5, 'C2': 2, 'gamma': 0.125})
        
        self.assertEqual(ovotsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_tsvm_get_param_names(self):
        
        """
        It checks the names of the hyper-parameters of TSVM estimator that returned
        by its get_param_names
        """
        
        self.assertEqual([TSVM('linear').get_params_names(),
                          TSVM('RBF').get_params_names()],
                          [['C1', 'C2'], ['C1', 'C2', 'gamma']])
        
    def test_mctsvm_get_param_names(self):
        
        """
        It checks the names of the hyper-parameters of MCTSVM estimator that returned
        by its get_param_names
        """
        
        self.assertEqual([MCTSVM('linear').get_params_names(),
                          MCTSVM('RBF').get_params_names()],
                          [['C'], ['C', 'gamma']])
        
    def test_ovotsvm_get_param_names(self):
        
        """
        It checks the names of the hyper-parameters of OVOTSVM estimator that returned
        by its get_param_names
        """
        
        self.assertEqual([OVO_TSVM('linear').get_params_names(),
                          OVO_TSVM('RBF').get_params_names()],
                          [['C1', 'C2'], ['C1', 'C2', 'gamma']])
        
        

if __name__ == '__main__':

    unittest.main()