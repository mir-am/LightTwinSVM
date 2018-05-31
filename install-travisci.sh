#!/bin/bash

# LightTwinSVM Program - Simple and Fast
# Version: 0.1.0-alpha - 2018-05-24
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# In this shell script, required dependencis will be installed.
# for building and testing LightTwinSVM on Travis CI

pip install -r requirments.txt

# clones Armadillo which is a C++ Linear Algebra library
# Armadillo is licensed under the Apache License, Version 2.0
git clone -b 8.500.x --single-branch https://github.com/conradsnicta/armadillo-code.git temp

# Lookslike there is a bug in python-config which produces wrong extension suffix
# More info at https://bugs.python.org/issue25440
# Following solution solves the problem on Travis CI
# Get extension suffix for building Python extension module
ext_suffix='./src/clippdcd'$(python3 -c $'from distutils.sysconfig import get_config_var; print(get_config_var(\'EXT_SUFFIX\'))')

# Building C++ extension module
c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./src/optimizer/pybind_clippdcd.cpp -o $ext_suffix -I ./temp/include -DARMA_DONT_USE_WRAPPER -lblas -llapack 

# Creates result directory for saving unit test's output
mkdir "result"
