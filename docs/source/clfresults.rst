Classification results
======================

The LightTwinSVM program saves classification results in an Excel file. Here, the description of each column of the Excel file for both binary and multi-class classification problems is given in below tables. It helps you analyze the classification results.

.. table :: The description of the Excel file for a binary classification problem.

	+-----------------+-----------------------------------------------------------------+
	| Column name     |                      Description                                |
	+=================+=================================================================+
	| accuracy        | The overall classification accuracy                             |
	+-----------------+-----------------------------------------------------------------+
	| acc_std         | The standard deviation of the overall classification accuracy   |
	+-----------------+-----------------------------------------------------------------+
	| recall_p        | The recall for the positive class                               |
	+-----------------+-----------------------------------------------------------------+
	|    r_p_std      | The standard deviation of the recall for the positive class     |
	+-----------------+-----------------------------------------------------------------+
	|    precision_p  | The precision for the positive class                            |
	+-----------------+-----------------------------------------------------------------+
	|    p_p_std      | The standard deviation of the precision for the positive class  |
	+-----------------+-----------------------------------------------------------------+
	|    f1_p         | The F1-measure for the positive class                           |
	+-----------------+-----------------------------------------------------------------+
	|    f1_p_std     | The standard deviation of the F1-measure for the positive class |
	+-----------------+-----------------------------------------------------------------+
	|    recall_n     | The recall for the negative class                               |
	+-----------------+-----------------------------------------------------------------+
	|    r_n_std      | The standard deviation of the recall for the negative class     |
	+-----------------+-----------------------------------------------------------------+
	|    precision_n  | The precision for the negative class                            |
	+-----------------+-----------------------------------------------------------------+
	|    p_n_std      | The standard deviation of the precision for the negative class  |
	+-----------------+-----------------------------------------------------------------+
	|    f1_n         | The  F1-measure for the negative class                          |
	+-----------------+-----------------------------------------------------------------+
	|    f1_n_std     | The standard deviation of the F1-measure for the negative class |
	+-----------------+-----------------------------------------------------------------+
	| tp              | True positive                                                   |
	+-----------------+-----------------------------------------------------------------+
	| tn              | True negative                                                   |
	+-----------------+-----------------------------------------------------------------+
	| fp              | False positive                                                  |
	+-----------------+-----------------------------------------------------------------+
	| fn              | False negative                                                  |
	+-----------------+-----------------------------------------------------------------+
	| C1              | The value of the first penalty parameter for TwinSVM            |
	+-----------------+-----------------------------------------------------------------+
	| C2              | The value of the second penalty parameter for TwinSVM           |
	+-----------------+-----------------------------------------------------------------+
	| gamma           | The value of RBF kernel's parameter                             |
	+-----------------+-----------------------------------------------------------------+


.. table :: The description of the Excel file for a multi-class classification problem.

	+---------------------+-------------------------------------------------------------------------+
	| Column name         | Description                                                             |
	+=====================+=========================================================================+
	|    accuracy         | The overall classification accuracy                                     |
	+---------------------+-------------------------------------------------------------------------+
	| acc_std             | The standard deviation of the overall classification accuracy           |
	+---------------------+-------------------------------------------------------------------------+
	|    micro_recall     | The micro-averaged recall for all classes                               |
	+---------------------+-------------------------------------------------------------------------+
	|    m_rec_std        | The standard deviation of the micro-averaged recall for all classes     |
	+---------------------+-------------------------------------------------------------------------+
	|    micro_precision  | The micro-averaged precision for all classes                            |
	+---------------------+-------------------------------------------------------------------------+
	|    m_prec_std       | The standard deviation of the micro-averaged precision for all classes  |
	+---------------------+-------------------------------------------------------------------------+
	|    mirco_f1         | The micro-averaged F1-measure for all classes                           |
	+---------------------+-------------------------------------------------------------------------+
	|    m_f1_std         | The standard deviation of the micro-averaged F1-measure for all classes |
	+---------------------+-------------------------------------------------------------------------+
	| C1                  | The value of the first penalty parameter for TwinSVM                    |
	+---------------------+-------------------------------------------------------------------------+
	| C2                  | The value of the second penalty parameter for TwinSVM                   |
	+---------------------+-------------------------------------------------------------------------+
	| gamma               | The value of RBF kernel's parameter                                     |
	+---------------------+-------------------------------------------------------------------------+
	