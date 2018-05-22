## Intro
LightTwinSVM is a simple and fast implementation of standard Twin Support Vector Machine. Anyone who is interested in machine learning and classification can use this program for their work/projects.
 
The main features of the program are the following:
- A simple console program for running TwinSVM classifier
- Fast optimization algorithm: The ClippDCD algorithm was improved and is implemented in C++ for solving optimization problems of TwinSVM.
- Both Linear and RBF kernel are supported.
- K-fold cross validation supported.
- Training/Test split supported.
- It supports grid search over C and gamma parameters.
- Detailed classification result will be saved in a spreadsheet file.

Twin Support Vector Machine classifier proposed by: <br />
Khemchandani, R., & Chandra, S. (2007). Twin support vector machines for pattern classification. IEEE Transactions on pattern analysis and machine intelligence, 29(5), 905-910.

The ClippDCD algorithm was proposed by: <br />
Peng, X., Chen, D., & Kong, L. (2014). A clipping dual coordinate descent algorithm for solving support vector machines. Knowledge-Based Systems, 71, 266-278.

## Installation Guide
Currently, supported operating systems are as follows:
- Debian-based Linux systems (Ubuntu 16.04, Ubuntu 17.10 and Linux Mint 18)
- RPM-based Linux systems (Fedora)

First of all, [Python](https://www.python.org/) 3.4 interpreter or newer is required. Python 3 is usually installed by default on most Linux distributions.
In order to build and run the program, the following Python packages are needed:
- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](http://scikit-learn.org/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/intro.html)

In order to build C++ extension module(Optimizer), the following tools and libraries are required:
- [GNU C++ Compiler](https://gcc.gnu.org/)
- [Armadillo C++ Linear Algebra Library](http://arma.sourceforge.net/)
- [LAPACK](http://www.netlib.org/lapack/) and [BLAS](http://www.netlib.org/blas/) Library

**A Linux shell is created to help users download required dependencies and install program automatically.** However, make sure that [Git](https://git-scm.com/) and GNU C++ compiler is installed on your system.

To install the program, open a terminal and execute the following commands:
```
git clone https://github.com/mir-am/LightTwinSVM.git
cd LightTwinSVM && ./setup.sh
```
If the installation was successful, you'd be asked to delete temporary directory for installation. You can also run unit tests to check functionalities of the program. Finally, a Linux shell "ltsvm.sh" is created to run the program.
After the successful installation, LightTwinSVM program should look like this in terminal: <br />
![alt text](https://raw.githubusercontent.com/mir-am/LightTwinSVM/misc/img/LightTwinSVM.png)

## User Guide
LightTwinSVM is a simple console application. It has 4 steps for doing classification. Each step is explained below: <br />
**Step 1:** Choose your dataset by pressing Enter key. A file dialog window will be shown to help you find and select your dataset. Currently, CSV files are supported. <br />
**Step 2:** Choose a kernel function between Linear and Gaussin(RBF). RBF kernel often produces better classification result but takes more time. <br />

## Dataset Format

## Numerical Experiments


