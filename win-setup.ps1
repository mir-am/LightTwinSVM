# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# A power shell script for generating pre-bulit Windows binary.

echo "Installation of LightTwinSVM on Windows.\nPlease make sure that Python 3.5 or newer and Visual Studio 2015 or newer is installed on your system."
[void] (Read-Host "Press enter to start the installation process...")

$startTime = (Get-Date)

# Step 1:
# Installing dependencies using pip
echo "Step 1: Installing dependencies for Python..."
pip install -r "requirments.txt" --user
pip install Cython --user
echo "Step 1 completed..."

cd .\ltsvm\optimizer

echo "Step 2: Need to clone Armadillo C++ library"

# Step 2:
if (Test-Path -Path ".\armadillo-code"){

    echo "Found armadillo repo. No need to clone again."

}
else{

    # clone Armadillo which is a C++ Linear Algebra library
    # Armadillo is licensed under the Apache License, Version 2.0
    git clone https://github.com/mir-am/armadillo-code.git

}

echo "Step 2 completed... (Cloned Armadillo Library)"

# Step 3:
echo "Step3: Generate C++ extension module (clipDCD optimizer)"
# Name of Python extension module
$py_ext = (python -c "from distutils.sysconfig import get_config_var; print(get_config_var('EXT_SUFFIX'))")

if (Test-Path -Path "clipdcd$py_ext"){
    
    echo "Found C++ extension module. No need to build again."

}else{

    # Generate C++ extension module (Optimizer) using Cython
    python setup.py build_ext --inplace

}

cd ..\..\

echo "Step 3 completed... (Generated extension module)"

# PyInstaller sometimes cannot generate a binary executable on all Systems
# So generating exe file for Windows is not available for a while.... until a workaround is found.
# Step 3:
#if (Test-Path -Path ".\dist\LightTwinSVM.exe"){

#    echo "Found pre-built Windows binary."

#}else{

    # Generate pre-bulit Windows binary using PyInstaller 3.4-dev
#    pyinstaller -F pyinstaller_spec.spec

#}

#echo "Step 3 completed... (pre-bulit Windows binary)"

# Step 4:
# Add BLAS and LAPACK libs to PATH to avoid import error: DLL load failed
echo "Step 4: Adding Libraries and program to the PATH...."

# A batch script for running LightTwinSVM in CMD
Set-Content -Path "ltsvm.bat" -Value "python -m ltsvm"

$currentPath = (Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH).Path
$currentDir = (Get-Item -Path ".\").FullName
$libPath = Join-Path $currentDir "ltsvm\optimizer\armadillo-code\lib_win64\"
#$newPath = $currentDir + ';' + $libPath
$updatedPath = $currentPath + $libPath
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $updatedPath

# Add libs to temporary env path in PowerShell so that 
# the program can be launched in the current session
$env:Path += ';' + $libPath

echo "Step 4 completed"

$elapsedTime = (((Get-Date) - $startTime).TotalSeconds).ToString("0.000")
echo "The installation finished in $elapsedTime seconds."
echo "To launch the program, run ltsvm.bat in the root of LightTwinSVM project using PowerShell or CMD."

# Ask users to run unit tests
$confirm = Read-Host "Would you like to run unit tests for checking program installation? [y/n]"

if($confirm -eq 'y'){

   python -m unittest discover -s tests

}

#cd .\dist

# & ".\LightTwinSVM.exe"
