# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# A power shell script for generating pre-bulit Windows binary.

$startTime = (Get-Date)

# Installing dependencies using pip
pip install -r "requirments.txt" --user

cd .\src\optimizer

# Step 1:
if (Test-Path -Path ".\armadillo-code"){

    echo "Found armadillo repo. No need to clone again."

}
else{

    # clone Armadillo which is a C++ Linear Algebra library
    # Armadillo is licensed under the Apache License, Version 2.0
    git clone https://github.com/mir-am/armadillo-code.git

}

echo "Step 1 completed... (Cloned Armadillo Library)"

# Step 2:
# Name of Python extension module
$py_ext = (python -c "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))")

if (Test-Path -Path "..\clippdcd$py_ext"){
    
    echo "Found C++ extension module. No need to build again."

}else{

    # Generate C++ extension module (Optimizer) using Cython
    python setup.py build_ext --inplace

    mv "clippdcd$py_ext" ..\

}

cd ..\

echo "Step 2 completed... (Generated extension module)"

# Step 3:
if (Test-Path -Path ".\dist\LightTwinSVM.exe"){

    echo "Found pre-built Windows binary."

}else{

    # Generate pre-bulit Windows binary using PyInstaller 3.4-dev
    pyinstaller -F pyinstaller_spec.spec

}

echo "Step 3 completed... (pre-bulit Windows binary)"

$elapsedTime = (((Get-Date) - $startTime).TotalSeconds).ToString("0.000")
echo "The installation finished in $elapsedTime seconds."

cd .\dist

& ".\LightTwinSVM.exe"
