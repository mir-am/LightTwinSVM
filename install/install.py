#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.1 Alpha (May 9, 2018)
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0n

This module is created for installation of this program.
It checks Python interpreter and program's dependencies.

"""


import sys
import pip

def install_package(package_name):

	"""
		It installs required package with pip tool.
		Input:
		     package_name: Name of package to install.
	"""

	pip.main(['install', package_name])


# Gets exact version of user's Python interpreter
py_intp_info = sys.version_info
py_ver_major = py_intp_info.major
py_ver_minor = py_intp_info.minor

print("Python %d.%d detected on your system..." % (py_ver_major, py_ver_minor))
print("***************************************\n")

print("Step 2:\nChecks Python dependencies to run the program...")

# Checks requirements to run the program
# Follwing packages is needed
packages = ('numpy', 'pandas', 'scikit-learn', 'scipy', 'pybind11')

# A list of installed Python packages is created by shell script
module_file = open('./install/usermodules.txt', 'r')
list_modules = module_file.readlines()

user_packages = {}

for module in list_modules:
    
    module_name, version = tuple(module.split('=='))
    
    user_packages[module_name] = version.rstrip()

for p in packages:
    
    if p in user_packages:
        
        print("%s (ver. %s) is installed." % (p, user_packages[p]))
        
    else:
        
        print("Package %s is NOT installed. It will be installed by pip." % (p))
        install_package(p)

        
