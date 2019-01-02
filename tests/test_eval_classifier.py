#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightTwinSVM Program - Simple and Fast
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

This test module checks the functionalties of eval_classifier
"""

# A temprory workaround to import LightTwinSVM for running tests
#import sys
#sys.path.append('./')

from ltsvm.eval_classifier import search_space
import unittest


class TestEvalClassifier(unittest.TestCase):
    
    """
        it tests the utility functions and classes of eval_classifier.py module
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def test_search_space_bin_ovo_linear(self):
        
        """
        It tests creating search space for Binary and OVO estimator (Linear)
        """
        
        expected_output = [{'C1': 0.5, 'gamma': 1, 'C2': 0.5},
                           {'C1': 0.5, 'gamma': 1, 'C2': 1.0},
                           {'C1': 0.5, 'gamma': 1, 'C2': 2.0},
                           {'C1': 1.0, 'gamma': 1, 'C2': 0.5},
                           {'C1': 1.0, 'gamma': 1, 'C2': 1.0},
                           {'C1': 1.0, 'gamma': 1, 'C2': 2.0},
                           {'C1': 2.0, 'gamma': 1, 'C2': 0.5},
                           {'C1': 2.0, 'gamma': 1, 'C2': 1.0},
                           {'C1': 2.0, 'gamma': 1, 'C2': 2.0}]

        
        params = search_space('linear', 'binary', -1, 1, None, None)
        
        self.assertEqual(params, expected_output, "Grid of elements doesn't match")
        
        
    def test_search_space_bin_ovo_rbf(self):
        
        """
        It tests creating search space for Binary and OVO estimator (RBF)
        """
        
        expected_output = [{'C2': 0.5, 'C1': 0.5, 'gamma': 1.0},
                           {'C2': 0.5, 'C1': 0.5, 'gamma': 2.0},
                           {'C2': 1.0, 'C1': 0.5, 'gamma': 1.0},
                           {'C2': 1.0, 'C1': 0.5, 'gamma': 2.0},
                           {'C2': 2.0, 'C1': 0.5, 'gamma': 1.0},
                           {'C2': 2.0, 'C1': 0.5, 'gamma': 2.0},
                           {'C2': 0.5, 'C1': 1.0, 'gamma': 1.0},
                           {'C2': 0.5, 'C1': 1.0, 'gamma': 2.0},
                           {'C2': 1.0, 'C1': 1.0, 'gamma': 1.0},
                           {'C2': 1.0, 'C1': 1.0, 'gamma': 2.0},
                           {'C2': 2.0, 'C1': 1.0, 'gamma': 1.0},
                           {'C2': 2.0, 'C1': 1.0, 'gamma': 2.0},
                           {'C2': 0.5, 'C1': 2.0, 'gamma': 1.0},
                           {'C2': 0.5, 'C1': 2.0, 'gamma': 2.0},
                           {'C2': 1.0, 'C1': 2.0, 'gamma': 1.0},
                           {'C2': 1.0, 'C1': 2.0, 'gamma': 2.0},
                           {'C2': 2.0, 'C1': 2.0, 'gamma': 1.0},
                           {'C2': 2.0, 'C1': 2.0, 'gamma': 2.0}]

        
        
        params = search_space('RBF', 'binary', -1, 1, 0, 1)
        
        self.assertEqual(params, expected_output, "Grid of elements doesn't match")
        
    def test_search_space_ova_linear(self):
        
        """
        It tests creating search space for OVA estimator (linear)
        """
        
        exptected_output = [{'C': 0.5, 'gamma': 1},
                            {'C': 1.0, 'gamma': 1},
                            {'C': 2.0, 'gamma': 1}]
        
        params = search_space('linear', 'ova', -1, 1, None, None)
        
        self.assertEqual(params, exptected_output, "Grid of elements doesn't match")
        
    def test_search_space_ova_rbf(self):
        
        """
        It tests creating search space for OVA estimator (RBF)
        """
        
        expected_output = [{'C': 0.5, 'gamma': 1.0},
                           {'C': 0.5, 'gamma': 2.0},
                           {'C': 1.0, 'gamma': 1.0},
                           {'C': 1.0, 'gamma': 2.0},
                           {'C': 2.0, 'gamma': 1.0},
                           {'C': 2.0, 'gamma': 2.0}]
        
        params = search_space('RBF', 'ova', -1, 1, 0, 1)
        
        self.assertEqual(params, expected_output, "Grid of elements doesn't match")
        
        
if __name__ == '__main__':

    unittest.main()