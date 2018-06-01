# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# A power shell script for generating pre-bulit Windows binary.

cd .\src\optimizer

# Step 1:
# clone Armadillo which is a C++ Linear Algebra library
# Armadillo is licensed under the Apache License, Version 2.0
git clone -b 8.500.x --single-branch https://github.com/conradsnicta/armadillo-code.git

echo "Step 1 completed... (Cloned Armadillo Library)"

# Step 2: 
# Generate C++ extension module (Optimizer) using Cython
python setup.py build_ext --inplace

echo "Step 2 completed... (Generated extension module)"

# Name of Python extension module
$py_ext = (python -c "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))")

#mv "clippdcd$py_ext" ..\
cd ..\

# Step 3:
# Generate pre-bulit Windows binary using PyInstaller 3.4-dev
pyinstaller -F pyinstaller_spec.spec

echo "Step 3 completed... (pre-bulit Windows binary)"

cd .\dist
pwd

& ".\LightTwinSVM.exe"
