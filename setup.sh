#!/bin/bash

# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# This shell script helps users install the denependencies and test the program.
# Counting success of each step. Installation process consists of 7 steps.
step=0
# Creating a shell script for running the program
create_script()
{
arg=\${1-\"-n\"}

cat << EOM > "ltsvm.sh"
#!/bin/bash
case $arg in
    "-n")
    python3 src/main.py # LightTwinSVM program
    ;;
    "-t")
    python3 src/test_program.py # Unit tests
    ;;
    *)
    echo -e "Invalid argument.\n-n normal mode\n-t test mode"
esac
EOM
}

echo -e "The Installation of LightTwinSVM program:"
echo "Make sure that you are connected to the internet so that dependencies can be downloaded."
read -p "Press enter to start the installation process..."

echo -e "***************************************\nStep 1:"

start=$(date +%s)

# Detecting Python3 interpreter on user's Linux system.
py_intp=$(python3 -c 'import sys; sys.exit(0 if sys.version_info[0] == 3 and sys.version_info[1] >= 4 else 1)')

if [ $? == 0 ]
then

	echo -e "Python 3.4 or newer detected on your system..."
	((step++))
	
	# Check wether pip is installed.
	pip_check=$(which pip3)
	
	if [ $? == 0 ]
	then
		echo "pip tool is detected..."
		
		((step++))
	else

		echo "Could not detect pip tool and will be installed. "
		sudo apt-get install python3-pip -y
		
		((step++))
	fi
	
	# Installing Python packages by pip tool for users
	echo "***************************************"
	echo -e "Step 2:\nInstalls required Python packages to run the program..."
	
	pip3 install -r "requirments.txt" --user
	
	((step++))
	
	echo "Checks existence of Python's Tkinter..."
	
	py_tk=$(python3 -c 'import tkinter')
	
	if [ $? == 0 ]
	then
		echo "Found python3-tk on your system..."
		
		((step++))
	else
		echo "Could not find Tkinter and will be installed."
		
		if [[ ! -z $(which apt-get) ]];
		then
			sudo apt-get install python3-tk -y

		elif [[ ! -z  $(which yum) ]];
		then
			sudo yum install python3-tkinter -y
		fi
		
		((step++))
	fi

	echo "Looking for Python 3 dev. package..."
	
	py_dev=$(which python3-config)

	if [ $? == 0 ]
	then
	    echo "Found Python 3 dev on your system..."

	    ((step++))
	else
	    echo "Could not find Python 3 dev and will be installed."
		
		if [[ ! -z $(which apt-get) ]];
		then
			sudo apt-get install python3-dev -y
			
		elif [[ ! -z  $(which yum) ]];
		then	
			sudo yum install python3-devel -y
		fi
		
		((step++))
	fi
	
	echo -e "***************************************\n"
	
	# In order to build C++ extension module, LAPACK and BLAS library should be present.
	echo -e "Step 3:\nChecks existence of LAPACK and BLAS..."

    	if [ "$OSTYPE" == "linux-gnu" ]
    	then

	    lapack=$(ldconfig -p | grep liblapack)

	    if [ $? == 0 ]
	    then
		echo "Found LAPACK library on your Linux system..."

		((step++))
	    else
		echo "Could not find LAPACK on your system..."

		if [[ ! -z $(which apt-get) ]];
		then
		    sudo apt-get install liblapack-dev libblas-dev -y
		else
		    sudo yum install lapack-devel blas-devel -y
		fi
                ((step++))

	    fi

    	elif [ "$OSTYPE" == "darwin"* ]
    	then
            echo "A MacOS system is detected. So Accelerate Framework will be used for compiling C++ extension."
            ((step++))
    	fi

	ext_module="src/clippdcd$(python3-config --extension-suffix)"

	if [ -e $ext_module ]
	then
		echo "Found ClippDCD optimizer (C++ extension module.)"
		
		((step++))
	else
			
		if [ -d "temp" ]
		then
			echo "Found Armadillo repository. No need to clone again."
			
		else
			# clones Armadillo which is a C++ Linear Algebra library
			# Armadillo is licensed under the Apache License, Version 2.0
			git clone https://github.com/mir-am/armadillo-code.git temp
		fi

        # Check OS type for compiling on Linux and MacOS systems
        if [ "$OSTYPE" == "linux-gnu" ]
        then
            # Compiles C++ extension module
	    g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./src/optimizer/pybind_clippdcd.cpp -o ./src/clippdcd`python3-config --extension-suffix` -I ./temp/include -DARMA_DONT_USE_WRAPPER -lblas -llapack

        elif [ "$OSTYPE" == "darwin"* ]
        then

	    g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` ./src/optimizer/pybind_clippdcd.cpp -o ./src/clippdcd`python3-config --extension-suffix` -I ./temp/include -DARMA_DONT_USE_WRAPPER -framework Accelerate

        fi
		
		echo "The C++ extension moudle is generated..."
		((step++))
	fi
	
	echo -e "***************************************\n"
	
	if [ $step -eq  7 ]
	then
	
		# Creates a directroy for saving detailed classification result
		if [ ! -d "result" ]
		then
			mkdir "result"
		fi
		
		if [ -e ltsvm.sh ]
		then
			echo -e "A shell script already created for running program.\nat this address:$(pwd)"
		else
		
			create_script
			
			chmod +x ltsvm.sh
			
			echo -e "A shell script \"ltsvm.sh\" is created to launch the LightTwinSVM program.\n at this address:$(pwd)"
		fi
		
		
		echo "The installation was successfully completed... "
		
		end=$(date +%s)
		runtime=$((end-start))
		echo "The installation finished in $runtime seconds."
		echo -e "***************************************\n"
		
		echo -e "Do you want to delete temp directory?It contains Armadillo library's git repository.[y/n]"
		read -p "" choice
		
		if [[ $choice =~ ^[Yy]$ ]]
		then
			rm -rf temp
			echo "The temp directory deleted!"
		fi
		
		echo -e "Do you want to run tests to make sure that the program works?\nIt takes several minutes.[y/n]"
		read -p "" test_choice
		
		if [[ $test_choice =~ ^[Yy]$ ]]
		then
			echo "Unit test started..."
			
			python3 ./src/test_program.py -v
		fi
		
		echo -e "***************************************\n"
		echo -e "To run the program, execute the shell script \"ltsvm.sh\" at this address:\n$(pwd)"
	
	else
		echo "The installation failed... Number of compeleted steps: $step"	
	fi

else
	echo "Could not detect Python 3 interpreter. Please install Python 3.4 or newer."
fi

