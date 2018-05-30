# LightTwinSVM

<h3>A simple, light-weight and fast implementation of standard Twin Support Vector Machine </h3>
<p align="center">
<a href="https://opensource.org/licenses/GPL-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg?style=flat-square" alt="License"></a>
<a href="https://travis-ci.org/mir-am/LightTwinSVM.svg?branch=master"><img src="https://api.travis-ci.org/mir-am/LightTwinSVM.svg?branch=master" alt="Travis-CI"></a>
</p>
<br />

1. [Introduction](#intro)
2. [Installation Guide](#installation-guide)
3. [User Guide](#user-guide)
4. [Dataset Format](#dataset-format)
5. [Support](#support)
6. [Numerical Experiments](#numerical-experiments)

## Intro
LightTwinSVM is a simple and fast implementation of standard Twin Support Vector Machine. It is licensed under the terms of GNU GPL v3. Anyone who is interested in machine learning and classification can use this program for their work/projects.
 
The main features of the program are the following:
- A simple console program for running TwinSVM classifier
- Fast optimization algorithm: The ClippDCD algorithm was improved and is implemented in C++ for solving optimization problems of TwinSVM.
- Both Linear and RBF kernel are supported.
- K-fold cross validation supported.
- Training/Test split supported.
- It supports grid search over C and gamma parameters.
- CSV and LIBSVM data files are supported.
- Detailed classification result will be saved in a spreadsheet file.

Twin Support Vector Machine classifier proposed by: <br />
Khemchandani, R., & Chandra, S. (2007). Twin support vector machines for pattern classification. IEEE Transactions on pattern analysis and machine intelligence, 29(5), 905-910.

The ClippDCD algorithm was proposed by: <br />
Peng, X., Chen, D., & Kong, L. (2014). A clipping dual coordinate descent algorithm for solving support vector machines. Knowledge-Based Systems, 71, 266-278.

## Installation Guide
Currently, supported operating systems are as follows:
- Debian-based Linux systems (Ubuntu 14.04, Ubuntu 16.04, Ubuntu 17.10 and Linux Mint 18)
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
![alt text](https://github.com/mir-am/LightTwinSVM/blob/misc/img/LightTwinSVM-dataset.png)<br />
**Step 2:** Choose a kernel function between Linear and Gaussin (RBF). RBF kernel often produces better classification result but takes more time.
<br />
```
Step 2/4: Choose a kernel function:(Just type the number. e.g 1)
1-Linear
2-RBF
-> 2
```
**Step 3:** To evaluate TwinSVM performance, You can either use [K-Fold cross validation](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f) or split your data into training and test sets. <br />
```
Step 3/4: Choose a test methodolgy:(Just type the number. e.g 1)
1-K-fold cross validation
2-Train/test split
-> 1
Determine number of folds for cross validaton: (e.g. 5)
-> 5
```
**Step 4:** You need to determine the range of C penalty parameter and gamma (If RBF kernel selected.) for exhaustive grid search. <br /> 
An example:
```
Step 4/4:Type the range of C penalty parameter for grid search:
(Two integer numbers separated by space. e.g. -> -5 5
-> -4 4
```
After completing the above steps, the exhaustive search will be started. When the search process is completed, a detailed classification result will be saved in a spreadsheet file. In this file, all the common evalaution metrics(e.g Accuracy, Recall, Precision and F1) are provided.<br />
A instace of spreadsheet file containing classification result can be seen [here](https://github.com/mir-am/LightTwinSVM/blob/misc/TSVM_RBF_5-F-CV_pima-indian_2018-05-23%2013:21.csv).
 

## Dataset Format
- **LIBSVM** data files are supported. Note that the extension of this file should be '*.libsvm'.
- For **comma separated value (CSV)** file, make sure that your dataset is consistent with the following rules:
1. First row can be header names. (It's optional.)
2. Frist column should be labels of samples. Moreover, labels of positive and negative samples should be 1 and -1, respectively.
3. All the values in dataset except headernames should be numerical. Nominal values are not allowed. <br />
To help you prepare your dataset and test the program, three datasets are inculded [here](https://github.com/mir-am/LightTwinSVM/tree/master/dataset).

## Support
**Have a question about the software?**<br />
You can contact me via [email](mailto:mir-am@hotmail.com). Feedbacks and suggestions for improvements are welcome.<br />

**Have a problem with the software or found a bug?**<br />
To let me know and fix it, please open an issue [here](https://github.com/mir-am/LightTwinSVM/issues). <br />
To report a problem or bug, please provide the following information:<br />
1. Error messages<br />
2. Output of the program.<br />
3. Explain how to produce the problem if possible.


## Numerical Experiments

