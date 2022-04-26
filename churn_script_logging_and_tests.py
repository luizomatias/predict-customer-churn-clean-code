"""
This file contains the tests functions to test churn functions inside
churn_libary.py

Autor: Luiz Otávio

Date: April 2022

"""

import os
import logging
import pandas as pd
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_target_create(create_target):
    '''
    test create target - See if column Attrition_Flag exist.
    '''
    try:
        assert 'Attrition_Flag' in df.columns
        logging.info("Testing create_target: SUCCESS")
    except KeyError as err:
        logging.error(
            "Testing create_target: Column Attrition_Flag does not exist")
        raise err

    try:
        df_with_target = create_target(df)
    except KeyError as err:
        logging.error(
            "Testing create_target: An error occurred during creating target column.")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function - test if the images are saved in the right folder
    '''
    try:
        assert os.path.isfile('images/eda/churn_distribution.png')
        assert os.path.isfile('images/eda/customer_age_distribution.png')
        assert os.path.isfile('images/eda/heatmap.png')
        assert os.path.isfile('images/eda/marital_status_distribution.png')
        assert os.path.isfile('images/eda/total_transaction_distribution.png')
        logging.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error("Testing perform_eda: error on checking images files.")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper: Testing if the function can run and the number of encoder columns.
    '''
    try:
        df_test_enconder = encoder_helper(df, cat_columns, '_Churn')
        logging.info("Testing test_encoder_helper: SUCCESS")
    except (AssertionError, NameError, KeyError) as err:
        logging.error(
            "Testing test_encoder_helper: Something went wrong in the function.")
        raise err

    try:
        df_test_enconder = encoder_helper(df, cat_columns, '_Churn')
        assert sum(df_test_enconder.columns.str.contains(
            '_Churn')) == len(cat_columns)
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: The length of the encoders columns  is its incorrect.")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering: testing the types of X_train, X_test, y_train and y_test
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, keep_cols)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_perform_feature_engineering: The type of some dataframe or series it's incorrect.")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(X, X_train, X_test, y_train, y_test)
        assert os.path.isfile('models/logistic_model.pkl')
        assert os.path.isfile('models/rfc_model.pkl')
        assert os.path.isfile('images/results/rf_results.png')
        assert os.path.isfile('images/results/logistic_results.png')
        assert os.path.isfile('images/results/roc_curve_result.png')
        assert os.path.isfile('images/results/feature_importances.png')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: Something went wrong with training function. Some file was not saved.")
        raise err


if __name__ == "__main__":

    test_import(cls.import_data)
    df = cls.import_data("./data/bank_data.csv")
    test_target_create(cls.create_target)
    df = cls.create_target(df)
    test_eda(cls.perform_eda(df))
    cat_columns = cls.cat_columns
    test_encoder_helper(cls.encoder_helper)
    keep_cols = cls.keep_cols
    test_perform_feature_engineering(cls.perform_feature_engineering)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df, keep_cols)
    X = df[keep_cols]
    test_train_models(cls.train_models)
