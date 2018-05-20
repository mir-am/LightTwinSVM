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
In order to build and run the program, the following Python packages are needed:
- [NumPy](www.numpy.org)
- [SciPy](https://www.scipy.org/)
- [Scikit-learn](http://scikit-learn.org/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
- [Pybind11](https://pybind11.readthedocs.io/en/stable/intro.html)


## User Guide


## Numerical Experiments


