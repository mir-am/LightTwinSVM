#!/bin/bash

# LightTwinSVM Program - Simple and Fast
# Version: 0.1.0-alpha - 2018-05-24
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# In this shell script, required dependencis will be installed.
# for building and testing LightTwinSVM on Travis CI

# For OS X, python can be installed using  brew on Travis CI
if [[ "$TRAVIS_OS_NAME" == "osx" ]]
then

#brew update # It's time-consuming
brew install pyenv || brew upgrade pyenv
brew install pyenv-virtualenv
pyenv install $PYENV_VERSION

# Manually adding to path!!
export PYENV_VERSION=$PYENV_VERSION
export PATH="/Users/travis/.pyenv/shims:${PATH}"
pyenv virtualenv venv
source venv/bin/activate

pip install --upgrade pip
python --version
pip --version
fi

pip install -r requirments.txt

# clones Armadillo which is a C++ Linear Algebra library
# Armadillo is licensed under the Apache License, Version 2.0
git clone https://github.com/mir-am/armadillo-code.git temp

# Building C++ extension module on Linux and OS X
if [[ "$TRAVIS_OS_NAME" == "linux" ]]
then

# Lookslike there is a bug in python-config which produces wrong extension suffix
# More info at https://bugs.python.org/issue25440
# Following solution solves the problem on Travis CI
# Get extension suffix for building Python extension module
ext_suffix='./ltsvm/optimizer/clipdcd'$(python3 -c $'from distutils.sysconfig import get_config_var; print(get_config_var(\'EXT_SUFFIX\'))')

g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./ltsvm/optimizer/pybind_clippdcd.cpp -o $ext_suffix -I ./temp/include -DARMA_DONT_USE_WRAPPER -lblas -llapack 

elif [[ "$TRAVIS_OS_NAME" == "osx" ]]
then
echo $(python-config --extension-suffix)
g++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python -m pybind11 --includes` ./ltsvm/optimizer/pybind_clippdcd.cpp -o ./ltsvm/optimizer/clipdcd`python-config --extension-suffix` -I ./temp/include -DARMA_DONT_USE_WRAPPER -framework Accelerate
fi

# Creates result directory for saving unit test's output
mkdir "result"

# For OSX, unit tests should be ran here. Because of virtualenv!
if [[ "$TRAVIS_OS_NAME" == "osx" ]]
then
python -m unittest discover -s tests
fi
