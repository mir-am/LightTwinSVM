# How to contribute to the LightTwinSVM
First of all, thank you for your interest in contributing. You can contribute to the LightTwinSVM project through [code](#code) or [documentation](#documentation).

## Code
You can contribute through code by creating a [new feature](#new-feature) or [fixing a bug](#bug-fix). I suggest you take a look at the [to-do list](#to-do-list), which have high priority and helps you get involved in the development of the LightTwinSVM project.

### New feature
Developing new features are very much appreciated and makes the LightTwinSVM a better tool. However, the main design goal is to keep LightTwinSVM **small** and **simple**. Some features might make the project complex. Therefore, If you want to develop a new feature, please create an issue with "new feature" label (An issue describes only one feature.). Before implementing the new feature, please describe the new feature by explaining why the new feature is needed, its requirements and what it does. Therefore, we discuss whether the new feature can be added to the LightTwinSVM or not.

After we agreed upon the development of the new feature, you should follow the [code guideline](#code-guideline) for developing the new feature.

### Bug fix
If you have found a bug, please read the support section for how to report a bug, then create an issue that describes the bug. You may want to fix the bug. If so, please follow the [code guideline](#code-guideline) for fixing a bug.

### To-do list
This is a list of features, which have high priority for the LightTwinSVM project. You can work on one of the following features or a new one outside the list. Either way, please read the guide on how to develop a [new future](#new-feature) for the project.

- A graphical user interface that has only one main window. It should be user-friendly and help users do all the steps (similar to the command-line interface) to solve a classification problem.
- A GPU version of the LightTwinSVM, which runs the computationally expensive operations on the GPU, such as matrix inversion, matrix multiplication, and possibly the optimizer.
- To assess and demonstrate the efficiency of the LightTwinSVM program, a simple benchmark is needed, which shows the train/test time of the TwinSVM classifier on a dataset. It may also show the user's system info such as OS and hardware specs.

### Code guideline
To develop a new feature or fix a bug, please do the following steps:
- Create a branch to identify the feature you'd like to work on (e.g. 1189-new-feature). In the case of a bug fix, similar to developing a new feature, create a new branch.
- start coding and develop the new feature or fix the bug. (Only one feature should be developed!)
- Make sure that the code follows the [PEP8](https://pep8.org/) style guide.
- Create unit tests that cover any changes you made. Make sure that all the tests (including yours) pass.
- Commit all your changes and create [a pull request](https://help.github.com/en/articles/creating-a-pull-request).
- I carefully review the pull request and changes. It may take several days or even several weeks. All in all, I will decide whether the new feature/bug fix should be added to master branch or not.
- By having an approved feature or a bug fix, you will become one of the contributors of the LightTwinSVM project.

## Documentation
To contribute through documentation, you can fix typo errors, or you may want to improve API docs by adding usage examples. For instance, you can create a new page in the [documentation](https://lighttwinsvm.readthedocs.io/en/latest/), which describes how to use the LightTwinSVM API to solve a classification task. As of writing this, the project's documentation lacks good usage examples which shows power users how to use TwinSVM estimators for their project/problem. Nonetheless, if you have a new idea on the improvement of the docs, let me know. (See the [support section](https://github.com/mir-am/LightTwinSVM#support) for contact details.)

### Guide
To improve the documentation, make your changes in the [docs](https://github.com/mir-am/LightTwinSVM/tree/master/docs) folder. Note that this project uses [Sphinx](http://sphinx-doc.org/) and its API doc follows the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) rules. After you made your changes, commit them, and then create [a pull request].