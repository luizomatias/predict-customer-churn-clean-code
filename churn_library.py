# library doc string
"""
This file contains the functions to the churn model.

Autor: Luiz Otávio

Date: April 2022

"""

# import libraries
import numpy as np
import joblib
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.use('Agg')
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)

    return df


def create_target(dataframe):
    '''
    returns dataframe with target Churn

    input:
            df: pandas dataframe
    output:
            df: pandas dataframe
    '''
    df = dataframe.copy()

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    plt.ioff()

    plt.figure(figsize=(20, 10))

    churn_distribution = df['Churn'].hist()
    churn_distribution.figure.savefig('images/eda/churn_distribution.png')

    customer_age_distribution = df['Customer_Age'].hist()
    customer_age_distribution.figure.savefig(
        'images/eda/customer_age_distribution.png')

    marital_status_distribution = df.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    marital_status_distribution.figure.savefig(
        'images/eda/marital_status_distribution.png')

    total_transaction_distribution = sns.histplot(
        df['Total_Trans_Ct'], stat='density', kde=True)
    total_transaction_distribution.figure.savefig(
        'images/eda/total_transaction_distribution.png')

    heatmap = sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    heatmap.figure.savefig('images/eda/heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat_columns in category_lst:
        groups = df.groupby(cat_columns).mean()['Churn']
        lst = [groups.loc[val] for val in df[cat_columns]]
        df[f'{cat_columns}{response}'] = lst

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    X_train, X_test, y_train, y_test = train_test_split(
        df[response], df['Churn'], test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    model_name = ['Random Forest', 'Logistic Regresson']
    y_train_models = [y_train_preds_rf, y_train_preds_lr]
    y_test_models = [y_test_preds_rf, y_test_preds_lr]
    name_image_save = ['rf_results', 'logistic_results']

    for model_index in range(0, 2):
        plt.rc('figure', figsize=(10, 10))
        plt.text(0.01, 1.25, str(f'{model_name[model_index]} Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_models[model_index])), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str(f'{model_name[model_index]} Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_models[model_index])), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig(f'images/results/{name_image_save[model_index]}.png')
        plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=75)
    plt.savefig(output_pth)
    plt.tight_layout()
    plt.show()
    plt.close()


def roc_curve_image(rfc_model, lr_model, X_test, y_test):
    '''
    create roc curve graph and save
    input:
              rfc_model: model object random forest
              lr_model: model object logistic regression
              X_test: X testing data
              y_test: y testing data
    output:
              None
    '''
    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test, ax=ax, alpha=0.8)
    ax.figure.savefig(f'images/results/roc_curve_result.png')
    plt.show()
    plt.close()


def train_models(X, X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X: X dataframe
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = (cv_rfc.best_estimator_.predict(X_train))

    y_test_preds_rf = (cv_rfc.best_estimator_.predict(X_test))

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    roc_curve_image(cv_rfc.best_estimator_, lrc, X_test, y_test)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X,
        f'images/results/feature_importances.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":

    dataframe = import_data("./data/bank_data.csv")
    dataframe = create_target(dataframe)
    perform_eda(dataframe)
    dataframe = encoder_helper(dataframe, cat_columns, '_Churn')
    features_train, features_test, target_train, target_test = perform_feature_engineering(
        dataframe, keep_cols)
    X = dataframe[keep_cols]
    train_models(X, features_train, features_test, target_train, target_test)
