#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: main.py

"""

#**********Warnings************
import warnings
# Silence warnings about NumPy
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#*******************************

from ui import program_ui


def main():
    
    program_ui()

  
if __name__ == '__main__':
    
    # The execution of the program starts from here.
    main()
    
