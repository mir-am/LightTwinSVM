# How to contribute to the LightTwinSVM
First of all, thank you for your interest in contributing. You can contribute to the LightTwinSVM project through code or documentation.

## Code
You can contribute through code by creating a [new feature](#new-feature) or [fixing a bug](#bug-fix). I suggest you to take a look at the [to-do list](#to-do-list), which helps you get involved in the development of the LightTwinSVM project.

### New feature
Developing new features are very much appreciated and makes the LightTwinSVM a better tool. However, the main design goal is to keep LightTwinSVM **small** and **simple**. Some features might make the project complex. Therefore, If you want to develop a new feature, please create an issue with "new feature" label (An issue describes only one feature.). Before implementing the new feature, please describe the new feature by explaining why the new feature is needed, its requirements and what it does. Hence we discuss about whether the new feature can be added to the LightTwinSVM or not.

After we agreed upon the development of the new feature, you should follow the [code guideline](#code-guideline) for developing the new feature.

### Bug fix
If you have found a bug, please read the support section for how to report a bug, then create an issue that describes the bug. You may want to fix the bug. If so, please follow the [code guideline](#code-guideline) for fixing a bug.

### To-do list
This is a list of features, which have high priority for the LightTwinSVM project. You can work on one of the following features or a new one outside the list. Either way, please read the guide on how to develop a [new future](#new-feature) for the project.

- A graphical user interface that has only one main window. It should be user-friendly and help users do all the steps (similar to command-line interface) to solve a classification problem.
- A GPU version of the LightTwinSVM, which runs the computationally expensive operations on the GPU, such as matrix inversion, matrix multiplication and the optimizer.
- To assess and demonstrate the efficiency of the LightTwinSVM program, a simple benchmark is needed, which shows the train/test time of the TwinSVM classifier on a dataset. It may also show the user's system info such as OS and hardware specs.

### Code guideline
To develop a new feature or fix a bug, please do the following steps:
- Create a branch to identify the feature you'd like to work on (e.g. 1189-new-feature)
- start coding and develop the new feature (Only one feature should be developed!)
- Make sure that the code follows the [PEP8](https://pep8.org/) style guide.
- Create unit tests that cover any changes you made. Make sure that all the tests (including yours) passes.
- Commit all your changes and create [a pull request](https://help.github.com/en/articles/creating-a-pull-request).
- I carefully review the pull request and changes. It may takes several days or even several weeks. All in all, I will decide whether the new feature should be added to master branch or not.
- By having an approved feature or bug fix, you will become one of the contributors of the LightTwinSVM project.

## Documentation
