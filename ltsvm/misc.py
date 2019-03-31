#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LightTwinSVM Program - Simple and Fast
# Version: 0.6.0 - 2019-03-31
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

"""
In this module, several miscellaneous functions are defined for using
in other module, such as date time formatting and customized progress bar.
"""

import itertools
import numpy as np

try:
    
    import matplotlib.pyplot as plt
    
    def plt_confusion_matrix(cm, classes, title='Confusion matrix',
                         cmap=plt.cm.Blues):
    
        """
        It plots a confusion matrix for a given 2d-array.
        
        Parameters
        ----------
        cm : array-like, shape (n_samples, n_features)
             The elements of the confusion matrix.
            
        classes : array-like, shape (n_samples,) 
            Unique class labels.    
        
        title : str 
            Title of the confusion matrix.
            
        cmap : object
            Colormap for confusion matrix. Its default value is blue.
        """
        
        # Normalizing confusion matrix
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes, rotation=90)
        
        fmt = '.2f'
        thresh = cm.max() / 2.0
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", \
                     color="white" if cm[i, j] > thresh else "black")
            
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
    
except ImportError:
    print("Couldn't import matplotlib package. However, this package is optional"
          " and required for plotting confusion matrix.")


def time_fmt(t_delta):
    
    """
    It converts datetime objects to formatted string.
    
    Parameters
    ----------
    t_delta : object
        The difference between two dates or time.
        
    Returns
    -------
    str
        A readable formatted-datetime string.
    """
   
    hours, remainder = divmod(t_delta, 3600)
    minutes, seconds = divmod(remainder, 60)
   
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def progress_bar_gs(iteration, total, e_time, accuracy, best_acc, prefix='', \
                    suffix='', decimals=1, length=25, fill='#'):
    """
    It shows a customizable progress bar for grid search.
    
    Parameters
    ----------
    iteration : int
        Current iteration.
    
    total : int
        Maximumn number of iterations.
        
    e_time : str
        Elapsed time.

    accuracy : tuple
        The accuracy and its std at current iteration (acc, std).
        
    best_acc : tuple 
        The best accuracy and its std that were obtained at current iteration
        (best_acc, std).
        
    prefix : str, optional (default='') 
        Prefix string.
        
    suffix : str, optional (default='') 
        Suffix string.
        
    decimals : int, optinal (default=1)
        Number of decimal places for percentage of completion.
        
    length : int, optional (default=25) 
        Character length of the progress bar.
    
    fill : str, optional (default='#') 
        Bar fill character.
    """ 
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    fill_length = int(length * iteration // total)
    bar = fill * fill_length + '-' * (length - fill_length)

    output = "\r%sB-Acc:%.2f+-%.2f|Acc:%.2f+-%.2f |%s| %s%% %sElapsed:%s"
    print(output % (prefix, best_acc[0], best_acc[1], accuracy[0], accuracy[1], \
                    bar, percent, suffix, e_time), end='\r')

    if iteration == total:
        print()	
