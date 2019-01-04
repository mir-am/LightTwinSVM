# Changelog
All notable changes to LightTwinSVM program will be documented in this file. <br />
Main contributor: [Mir, A.](https://github.com/mir-am) (mir-am@hotmail.com) <br />
License: [GNU General Public License v3.0](https://github.com/mir-am/LightTwinSVM/blob/master/LICENSE.txt)<br />

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2019-01-01
### Added
- The multi-class method of One-vs-All (OVA) was implemented.
- The multi-class method of One-vs-One (OVO) was implemented.
- The OVO estimator is scikit-learn compatible classifier.
- An option was added to command-line interface to choose between OVA and OVO schemes.
- LightTwinSVM is now Python package and its estimators and utilities can be imported in other Python projects.
- [API] get_param_names was added to esitmators' classes. It returns the hyper-parameters of the estimator.

### Changed
- Search space is now created using ParameterGrid of Scikit-learn.
- Validator class returns classification results as a Python dictionary.

### Fixed
- Wrong test size for Train/Test split was fixed.

## [0.4.0-alpha] - 2018-06-08
### Added
- Experimental support for Rectangular kenrnel.

### Changed
- eval_classfier.py, twinsvm.py and ui.py modules are re-written and improved.

## [0.3.0-alpha] - 2018-06-03
### Added
- Experimental support for Windows OS.

## [0.2.0-alpha] - 2018-05-30 
### Added
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) data is now supported.
- A customized progress bar for grid search

## [0.1.0-alpha] - 2018-05-24
### Added
- A simple console app created.
- Fast Optimization algorithm (ClippDCD optimizer)
- Both Linear and RBF kernel supported.
- K-fold cross validation supported.
- Training/Test split supported.
- Support for exhaustive grid search over C and gamma parameters.
- Detailed classification result in a spreadsheet file.
