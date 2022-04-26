# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict churn according to customers credit card. The project contains modular functions, best practices according to PEP8 using pylint. Unit testing and logging are including too. 

## Files and data description
The project have the following structure:  


├── churn_library.py        --------> This file contains the functions to the churn model
├── churn_notebook.ipynb    --------> Jupyter notebook from churn model 
├── churn_script_logging_and_tests.py  ------> This file contains the tests functions to test churn functions
├── data
│   └── bank_data.csv   ------> data in csv format
├── images
│   ├── eda 
│   │   ├── churn_distribution.png  ------> churn distribution image
│   │   ├── customer_age_distribution.png  ------> customer age distribution image
│   │   ├── heatmap.png  ------> heatmap image
│   │   ├── marital_status_distribution.png  ------> marital status distribution image
│   │   └── total_transaction_distribution.png  ------> total transaction distribution
│   └── results
│       ├── feature_importances.png  ------> feature importances image
│       ├── logistic_results.png  ------> logistic model results image
│       ├── rf_results.png  ------> random forest results image
│       └── roc_curve_result.png  ------> roc curve result image
├── logs
│   └── churn_library.log  ------> test logging
├── models
│   ├── logistic_model.pkl ------> logistic model file
│   └── rfc_model.pkl ------> random forest file
│
├── README.md
├── requirements_py3.6.txt

## Running Files
You can use ipython or python to run the files. You can run the model with:

```
ipython churn_library.py
```
or 

```
python churn_library.py
```

If everything runs perfectly, the models built will be created in models folder. 
And some eda images and results metrics from model will be created in images/eda 
and images/results folders respectively.  

To run unit testing: 

```
ipython churn_script_logging_and_tests.py
```
or 

```
python churn_script_logging_and_tests.py
```

All logs from unit test will be saved in churn_library.log file inside logs folder.