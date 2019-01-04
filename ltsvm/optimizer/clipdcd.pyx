# cython: language_level=3
# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# A Cython module for wrapping C++ code (ClippDCD optimizer)
# It generates Python extension module for Windows OS.

from libcpp.vector cimport vector
import numpy as np
cimport numpy as np


cdef extern from "clippdcd_opt.h":
    
	vector[double] clippDCDOptimizer(vector[vector[double]] &dual, const double c)
	

def clippDCD_optimizer(np.ndarray[double, ndim=2] mat_dual, const double c):
	
	"""
	it calls C++ function
	Input:
	    - Dual matrix (NumPy 2d-array)
		- C penalty parameter (float)
	Output:
	    - Lagrange multipliers (List)
	"""

	return clippDCDOptimizer(mat_dual, c)
	