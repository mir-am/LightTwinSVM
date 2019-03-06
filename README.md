# LightTwinSVM

<h3>A simple, light-weight and fast implementation of standard Twin Support Vector Machine </h3>
<p align="center">
<a href="https://opensource.org/licenses/GPL-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg" alt="License"></a>
<a href=""><img src="https://img.shields.io/pypi/pyversions/Django.svg" alt="Python Versions"></a>
<a href="https://github.com/mir-am/LightTwinSVM/releases"><img src="https://img.shields.io/github/release/mir-am/LightTwinSVM/all.svg" alt="latest release version"></a>
<a href='https://lighttwinsvm.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/lighttwinsvm/badge/?version=latest' alt='Documentation Status' /></a>
<a href="https://travis-ci.org/mir-am/LightTwinSVM"><img src="https://api.travis-ci.org/mir-am/LightTwinSVM.svg?branch=master" alt="Travis-CI"></a>
<a href="https://ci.appveyor.com/project/mir-am/lighttwinsvm"><img src="https://ci.appveyor.com/api/projects/status/c625kl28aaqvuh9g?svg=true" alt="AppVeyor"></a>
</p>
<br />

1. [Introduction](#intro)
2. [Installation Guide](#installation-guide)
   - [Setup script](#setup-script-recommended)
   - [Building manually](#building-manually)
3. [User Guide](#user-guide)
   - [Usage example](#an-exmaple-of-using-command-line-interface)
   - [Tutorials](#tutorials)
   - [API documentation](#api-documentation)
4. [Dataset Format](#dataset-format)
5. [Support](#support)
6. [Contributing](#contributing) 
7. [FAQ](#frequently-asked-questions)
8. [Donations](#donations)
9. [Numerical Experiments](#numerical-experiments) 

## Intro
LightTwinSVM is a simple and fast implementation of standard Twin Support Vector Machine. It is licensed under the terms of GNU GPL v3. Anyone who is interested in machine learning and classification can use this program for their work/projects.
 
The main features of the program are the following:
- A **simple console program** for running TwinSVM classifier
- **Fast optimization algorithm:** The clipDCD algorithm was improved and is implemented in C++ for solving optimization problems of TwinSVM.
- **Linear**, **RBF** kernel and Rectangular are supported.
- Binary and **Multi-class classification** (One-vs-All & One-vs-One) are supported.
- The OVO estimator is **compatible with scikit-learn** tools such as GridSearchCV, cross_val_score, etc.
- The classifier can be evaluated using either **K-fold cross-validation** or **Training/Test** split.
- It supports **grid search** over C and gamma parameters.
- **CSV** and **LIBSVM** data files are supported.
- Detailed classification result will be saved in a spreadsheet file.

Twin Support Vector Machine classifier was proposed by: <br />
Khemchandani, R., & Chandra, S. (2007). Twin support vector machines for pattern classification. IEEE Transactions on pattern analysis and machine intelligence, 29(5), 905-910.

The clipDCD algorithm was proposed by: <br />
Peng, X., Chen, D., & Kong, L. (2014). A clipping dual coordinate descent algorithm for solving support vector machines. Knowledge-Based Systems, 71, 266-278.

## Installation Guide
Currently, supported operating systems are as follows. Choose your OS from list below for detailed install instructions.
- [Debian-based Linux systems](#linux--mac-os-x) (Ubuntu 14.04, Ubuntu 16.04, Ubuntu 17.10, Ubuntu 18.04 and Linux Mint 18)
- [RPM-based Linux systems](#linux--mac-os-x) (Fedora)
- [Mac OSX](#linux--mac-os-x)
- [Microsoft Windows](#windows)

### Dependencies
First of all, [Python](https://www.python.org/) 3.5 interpreter or newer is required. Python 3 is usually installed by default on most Linux distributions.
In order to build and run the program, the following Python packages are needed:
- [NumPy](https://www.numpy.org)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](http://scikit-learn.org/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/intro.html)
- [Cython](https://cython.org/)(To build C++ extension module on Windows.)
- [PyInstaller](https://www.pyinstaller.org/)(To generate a binary executable for Windows platform.)

In order to build C++ extension module(Optimizer), the following tools and libraries are required:
- [GNU C++ Compiler](https://gcc.gnu.org/) (For Linux systems)
- [Apple XCode](https://developer.apple.com/xcode/) (For OSX systems)
- [Visual Studio](https://visualstudio.microsoft.com/) (For Windows systems)
- [Armadillo C++ Linear Algebra Library](http://arma.sourceforge.net/)
- [LAPACK](http://www.netlib.org/lapack/) and [BLAS](http://www.netlib.org/blas/) Library


### Setup script (Recommended)
### Linux & Mac OS X
**A shell script is created to help users download required dependencies and install program automatically.** However, make sure that [Git](https://git-scm.com/) and GNU C++ compiler is installed on your system.

**A note for MacOS users:** Make sure that [Apple XCode](https://developer.apple.com/xcode/) is installed on your system.

To install the program, open a terminal and execute the following commands:
```
git clone https://github.com/mir-am/LightTwinSVM.git
cd LightTwinSVM && ./setup.sh
```
If the installation was successful, you'd be asked to delete temporary directory for installation. You can also run unit tests to check functionalities of the program. Finally, a Linux shell "ltsvm.sh" is created to run the program.
After the successful installation, LightTwinSVM program should look like this in terminal: <br />
![alt text](https://raw.githubusercontent.com/mir-am/LightTwinSVM/misc/img/LightTwinSVM.png)

### Windows
First, download Git program from [here](https://git-scm.com/) if it's not installed on your system. Also, [**Visual Studio 2015**](https://visualstudio.microsoft.com/) or newer should be installed so that C++ extension module can be compiled.

Before proceeding further, make sure that all the required Python packages are installed. Dependencies are listed [here](#dependencies). To install the program on Windows, open a PowerShell terminal and run the following commands:
```
git clone https://github.com/mir-am/LightTwinSVM.git
cd LightTwinSVM && .\win-setup.ps1
```
When the installation is finished, a batch file "ltsvm.bat" will be created to run the program.

### Building manually
It is highly recommended to install the LightTwinSVM program automatically using the setup script. If for some reasons you still want to build the program manually, a step-by-step guide is provided [here](https://github.com/mir-am/LightTwinSVM/wiki/Building-the-LightTwinSVM-manually-on-Linux-and-OSX-systems) for Linux and OSX systems.

## User Guide
### An exmaple of using command line interface
LightTwinSVM is a simple console application. It has 4 steps for doing classification. Each step is explained below: <br />
**Step 1:** Choose your dataset by pressing Enter key. A file dialog window will be shown to help you find and select your dataset. CSV and LIBSVM files are supported. It is highly recommended to normalize your dataset. <br />
![alt text](https://github.com/mir-am/LightTwinSVM/blob/misc/img/LightTwinSVM-dataset.png)<br />
**Step 2:** Choose a kernel function among Linear, Gaussian (RBF) and Rectangular. RBF kernel often produces better classification result but takes more time. However if you want to use non-linear kernel and your dataset is large, then consider choosing Rectangular kernel.
<br />
```
Step 2/4: Choose a kernel function:(Just type the number. e.g 1)
1-Linear
2-RBF
3-RBF(Rectangular kernel)
-> 2
```
**Step 3:** To evaluate TwinSVM performance, You can either use [K-Fold cross validation](https://towardsdatascience.com/cross-validation-in-machine-learning-72924a69872f) or split your data into training and test sets. <br />
```
Step 3/4: Choose a test methodology:(Just type the number. e.g 1)
1-K-fold cross validation
2-Train/test split
-> 1
Determine number of folds for cross validation: (e.g. 5)
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
A instance of spreadsheet file containing classification result can be seen [here](https://github.com/mir-am/LightTwinSVM/blob/misc/TSVM_RBF_5-F-CV_pima-indian_2018-05-23%2013:21.csv).
 
### Tutorials
LightTwinSVM can be imported as a Python package in your project. Currently, a Jupyter notebook is avaliable [here](https://github.com/mir-am/LightTwinSVM/tree/master/docs/notebooks), which is "A Step-by-Step Guide on How to Use Multi-class TwinSVM".

To run the notebooks, make sure that Jupyter is installed on your system. If not, use the following command to install it:
```
pip3 install jupyter
```
For more details, check out [Jupyter documentation](https://jupyter.readthedocs.io/en/latest/index.html).

### API documentation
Aside from the program's command line interface, you may want to use the LightTwinSVM's Python package for your project. All you have to do is to copy-paste the "[ltsvm](https://github.com/mir-am/LightTwinSVM/tree/master/ltsvm)" folder (the **installed version**) into the root folder of your project. Next, you can import "ltsvm" package in a module of your interest.

You can read about the documentation of the LightTwinSVM's estimators and tools [here](https://lighttwinsvm.readthedocs.io/en/latest/index.html). 

## Dataset Format
- **LIBSVM** data files are supported. Note that the extension of this file should be '*.libsvm'.
- For **comma separated value (CSV)** file, make sure that your dataset is consistent with the following rules:
1. First row can be header names. (It's optional.)
2. First column should be labels of samples. Moreover, labels of positive and negative samples should be 1 and -1, respectively.
3. All the values in dataset except headernames should be numerical. Nominal values are not allowed. <br />
To help you prepare your dataset and test the program, three datasets are included [here](https://github.com/mir-am/LightTwinSVM/tree/master/dataset).

## Support
**Have a question about the software?**<br />
You can contact me via [email](mailto:mir-am@hotmail.com). Feedback and suggestions for improvements are welcome.<br />

**Have a problem with the software or found a bug?**<br />
To let me know and fix it, please open an issue [here](https://github.com/mir-am/LightTwinSVM/issues). <br />
To report a problem or bug, please provide the following information:<br />
1. Error messages<br />
2. Output of the program.<br />
3. Explain how to reproduce the problem if possible.

## Contributing
Thanks for considering contribution to the LightTwinSVM project. Contributions are highly appreciated and welcomed. For guidance on how to contribute to the LightTwinSVM project, please see the [contributing guideline](https://github.com/mir-am/LightTwinSVM/blob/master/CONTRIBUTING.md).

## Frequently Asked Questions
- What is the main idea of TwinSVM classifier? <br />
TwinSVM does classification by using two non-parallel hyperplanes as opposed to a single hyperplane in the standard SVM. In TwinSVM, each hyperplane is as close as possible to samples of its own class and far away from samples of other class. To know more about TwinSVM and its optimization problems, you can read [this blog post](https://mirblog.me/index.php/2018/12/07/a-brief-intro-to-twin-support-vector-machine-classifier/ "A brief Introduction to TwinSVM classifier").

## Donations
[![Donate](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/mir-am/)

If you have used the LightTwinSVM program and found it helpful, please consider making a donation via [Liberapay](https://liberapay.com/mir-am/) to support this work. It also motivates me to maintain and develop new features for the program.

## Numerical Experiments
In order to indicate the effectiveness of the LightTwinSVM in terms of accuracy, experiments were conducted to compare it with scikit-learn's SVM on several UCI benchmark datasets. Similar to most research papers on classification, K-fold cross-validation is used to evaluate these classifiers (K was set to 5). Also, grid search was used to find the optimal values of hyper-parameters. Table below shows the accuracy comparison between the LightTwinSVM and Scikit-learn's SVM. <br />

| Datasets  | LightTwinSVM | Scikit-learn's SVM | Difference in Accuracy |
| ------------- | ------------- | ------------- | ------------- |
| Pima-Indian  | **78.91** | 78.26 | 0.65 |
| Australian | **87.25** | 86.81 | 0.44 |
| Haberman  | 76.12 | **76.80** | -0.68 |
| Cleveland  | **85.14** | 84.82 | 0.32 |
| Sonar  | **75.16** | 64.42 | 10.74 |
| Heart-Statlog | **85.19** | **85.19** | 0 |
| Hepatitis | **85.81** |83.23 | 2.58 |
| WDBC | **98.24** |98.07 | 0.17 |
| Spectf | **80.55** |79.78 | 0.81 |
| Titanic | **82.04** |81.71 | 0.33 |
| Mean | **83.44** |81.90 | 1.53 |

From the above table, it can be found that LightTwinSVM is more efficient in terms of accuracy. Therefore, it outperforms Sklearn's SVM on most datasets. All in all, if you have used SVM for your task/project, the LightTwinSVM program may give you a better predication accuracy for your classification task. More information on this experiment can be found in [this blog post](https://mirblog.me/index.php/2018/12/28/a-accuracy-comparison-between-scikit-learns-svm-and-lighttwinsvm-program/).

## Acknowledgements
Thanks to [idejie](https://github.com/idejie) for test and support on the MacOS. (Dec 8, 2018)
