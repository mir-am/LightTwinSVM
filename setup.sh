#!/bin/bash

# LightTwinSVM Program - Simple and Fast
# Version: 0.1 Alpha (May 9, 2018)
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# This shell script helps users install the denependencies and test the program.


echo -e "The Installation of LightTwinSVM program:"
echo "Make sure that you are connected to the internet so that dependencies can be downloaded."
read -p "Press enter to start the installation process..."

# Root previleges for installing dependencies.
#sudo apt-get update

echo -e "***************************************\nStep 1:"

start=$(date +%s)

# Detecting Python3 interpreter on User's linux system.
py_intp=$(which python3)

if [ $? == 0 ]
then

	py_ver="$(basename "${py_intp}")"

	# Check wether pip is installed.
	pip_check=$(which pip3)
	
	if [ $? == 0 ]
	then
		
		echo "pip tool is detected..."
		
	else

		echo "Could not detect pip tool and will be installed. "
		sudo apt-get install python3-pip -y

	fi
	
	# List of installed Python packages
	pip3 freeze | cat >> ./install/usermodules.txt

	echo "A list of installed Python packages is created..."

	if [ $py_ver == "python3" ]
	then
		python3 ./install/install.py
	fi
	
	echo "Checks existence of python's Tkinter..."
	
	py_tk=$(locate python3-tk)
	
	if [ $? == 0 ]
	then
		echo "Found python3-tk on your system..."
	else
		echo "Could not find Tkinter and will be installed."
		sudo apt-get install python3-tk -y
	fi
	
	echo -e "***************************************\n"
	
	# In order to build C++ extension module, LAPACK and BLAS library should be present.
	echo -e "Step 3:\nChecks existence of LAPACK and BLAS..."

	lapack=$(ldconfig -p | grep liblapack)

	if [ $? == 0 ]
	then
		echo "Found LAPACK library on your system..."
	else
		echo "Could not find LAPACK on your system..."
		sudo apt-get install liblapack-dev libblas-dev -y
	fi
	
	ext_module="src/clippdcd$(python3-config --extension-suffix)"
	
	if [ -e $ext_module ]
	then
		echo "Found ClippDCD optimizer (C++ extension module.)"
	else
	
		cd install
		
		if [ -d "$./armadillo-code" ]
		then
			echo "Found Armadillo repository. No need to clone again."
			
		else
			# clones Armadillo which is a C++ Linear Algebra library
			# Armadillo is licensed under the Apache License, Version 2.0
			git clone -b 8.500.x --single-branch https://github.com/conradsnicta/armadillo-code.git
		fi
		
		# Compiles C++ extension module
		c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ../src/optimizer/clippdcd.cpp -o ../src/clippdcd`python3-config --extension-suffix` -I ./armadillo-code/include -DARMA_DONT_USE_WRAPPER -lblas -llapack 
		
		cd ..
	
	fi
	
	echo -e "***************************************\n"
	
	if [ -e ltsvm.sh ]
	then
		echo -e "A shell script already created for running program.\nat this address:$(pwd)"
	else
	
		echo "#!/bin/bash
python3 src/main.py" >> ltsvm.sh
		
		chmod +x ltsvm.sh
		
		echo -e "A shell script \"ltsvm.sh\" is created to launch the LightTwinSVM program.\n at this address:$(pwd)"
	fi
	
	
	echo "The installation was successfully completed... "
	
	end=$(date +%s)
	runtime=$((end-start))
	echo "The installation finished in $runtime seconds."
	echo -e "***************************************\n"
	
	echo -e "Do you want to delete the temporary install directory?It contains install script and Armadillo library's git repository.[y/n]"
	read -p "" choice
	
	if [[ $choice =~ ^[Yy]$ ]]
	then
		rm -rf install2
		echo "The install temporary directory deleted!"
	fi
	
	echo -e "Do you want to run tests to make sure that the program works?\nIt takes several minutes.[y/n]"
	read -p "" test_choice
	
	if [[ $test_choice =~ ^[Yy]$ ]]
	then
		echo "Unit test started..."
		
		python3 ./src/test_program.py -v
	fi
	
	echo -e "***************************************\n"
	echo -e "To run the program, Execute the shell script \"ltsvm.sh\" at this address:\n$(pwd)"
	
	#blas=$(ldconfig -p | grep libblas)

	#if [ $? == 0 ]
	#then
	#	echo "Found BLAS library on your system..."
	#else
	#	echo "Could not find BLAS on your system..."
	#fi

else

	echo "Could not detect Python 3 interpreter. Please install Python 3."
fi

