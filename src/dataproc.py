# -*- coding: utf-8 -*-


"""
LightTwinSVM Program - Simple and Fast
Version: 0.1 Alpha (May 9, 2018)
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: dataproc.py
In this module, functions for reading and processing datasets are defined.

"""


from os.path import splitext, split
import numpy as np
import csv


def conv_str_fl(data):
    
    """
        It converts string data to float for computation.
    
    """
    
    temp_data = np.zeros(data.shape)
    
    # Read rows
    for i in range(data.shape[0]):
        
        # Read coloums
        for j in range(data.shape[1]):
            
            temp_data[i][j] = float(data[i][j])
            
    return temp_data


def read_data(filename, ignore_header=False):
    
    """
        It converts CSV file to NumPy arrays for further operations like training
        
        Input:
            file_name: Path to the dataset file
            ignore_header: Ignoring first row of dataset because of header names
            
        output:
            data_samples: Training samples in NumPy array
            data_labels: labels of samples in NumPy array
            file_name: Name of dataset
    
    """
    

    data = open(filename, 'r')
    
    data_csv = csv.reader(data, delimiter=',')
    
    # Ignore hedaer names
    if ignore_header:
        
        data_array = np.array(list(data_csv))
        
    else:
        
        data_array = np.array(list(data_csv)[1:]) # [1:] for removing headers
    
    #Close file
    data.close()
    
    # Shuffle data
    #np.random.shuffle(data_array)                        
    
    # Convers string data to float
    data_train = conv_str_fl(data_array[:, 1:])                     
                         
    data_labels = np.array([int(i) for i in data_array[:, 0]])
    
    file_name = splitext(split(filename)[-1])[0]
    
    return data_train, data_labels, file_name 


# Read dataset in LIBSVM format and convert to CSV
def libsvm_read(filename, out_file=None):
    
    # Open file
    data = open(filename, 'r')
    
    # Open CSV file
    data_csv = list(csv.reader(data, delimiter=' '))
    
    # Append converted to a list - Only features
    convt_list = []
    
    # Append label of samples
    labels = []
    
    # number of features + class
    num_f = len(data_csv[0])
    
    # Intialiaze array
    #data_array = np.zeros((len(data_csv), num_f - 1), dtype=np.float)
    
    for i in range(len(data_csv)):
        
        # Store row in a list
        row_list = []
        
        # Append to label to row
        row_list.append(data_csv[i][0])
                
        # Column
        for j in range(len(data_csv[i]) - 1):
            
            #print('(%d, %d)' % (i, j))
        
            # Get number
            row_list.append(data_csv[i][j + 1].split(':')[1])
            
                
        convt_list.append(row_list)
        print("Processed %d sample..." % i)        

           
    data.close()
    
    # Convert to csv file
    with open(out_file, 'w', newline='') as out_csv:
        
        wr = csv.writer(out_csv, quoting=csv.QUOTE_ALL)
        
        for i in range(len(convt_list)):
            
            wr.writerow(convt_list[i])
    
    
    return data_csv, convt_list, labels
    