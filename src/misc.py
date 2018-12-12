#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: misc.py
In this module, several miscellaneous functions are defined for using in other modules.
Such as date time formatting and customized progress bar
"""

import itertools
import numpy as np

try:
    
    import matplotlib.pyplot as plt
    
    def plt_confusion_matrix(cm, classes, title='Confusion matrix',
                         cmap=plt.cm.Blues):
    
        """
        This function plots a confusion matrix
        Input:
            cm: A 2-d numpy array that contains confusion matrix
            classes: Labels of classes
            title(str): Title of confusion matrix at the top
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
    It convets datetime objects to formatted string
    """
   
    hours, remainder = divmod(t_delta, 3600)
    minutes, seconds = divmod(remainder, 60)
   
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def progress_bar_gs(iteration, total, e_time, accuracy, best_acc, prefix='', \
                    suffix='', decimals=1, length=25, fill='#'):
    """
    A customized progress bar for grid search
    Input:
        iteration: currrent iteration
        total: total iteration
        e_time: Elapsed time
        accuracy: Current accuracy and its std (Tuple)
        best_acc: Best accuracy and its std (Tuple)
        prefix: prefix string
        suffix: suffix string
        decimals: number of decimals in percent
        length: character length of bar
        fill: bar fill character
    """ 
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    fill_length = int(length * iteration // total)
    bar = fill * fill_length + '-' * (length - fill_length)

    output = "\r%sB-Acc:%.2f+-%.2f|Acc:%.2f+-%.2f |%s| %s%% %sElapsed:%s"
    print(output % (prefix, best_acc[0], best_acc[1], accuracy[0], accuracy[1], \
                    bar, percent, suffix, e_time), end='\r')

    if iteration == total:
        print()	
