# library doc string


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        plt.figure(figsize=(20,10)) 
        df['Churn'].hist();
        plt.savefig('./images/churn.png')

        plt.figure(figsize=(20,10))
        df['Customer_Age'].hist();
        plt.savefig('./images/test.png')
        


def encoder_helper(df, category_lst, response='Churn'):
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
        for category in category_lst:
                category_list=[]
                category_groups = df.groupby(category).mean()['Churn']

                for val in df[category]:
                        category_list.append(category_groups.loc[val])
                
                df[category+'_'+response] = category_list
        return df


def perform_feature_engineering(df, response=['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']):
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
        y = df['Churn']
        X_data = pd.DataFrame()
        X_data[response] = df[response]
        model_dict = dict()
        X_data_2 = X_data
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size= 0.3, random_state=42)
        
        model_dict['X_train']=X_train
        model_dict['X_test']=X_test
        model_dict['y_train']=y_train
        model_dict['y_test']=y_test
        
        return (model_dict, X_data_2)

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                output_pth):
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

        plt.clf()
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1, str('Random Forest Train Results'), {
                'fontsize': 10}, fontproperties='monospace')
        
        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        
        plt.text(0.01, 0.6, str('Random Forest Test Results'), {
                'fontsize': 10}, fontproperties='monospace')
        
        plt.text(0.01, 0.3, str(classification_report(y_train, y_train_preds_rf)), {
                'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        
        plt.text(0.6, 1, str('Logistic Regression Train Results'), {
                'fontsize': 10}, fontproperties='monospace')
       
        plt.text(0.6, 0.7, str(classification_report(y_train, y_train_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        
        plt.text(0.6, 0.6, str('Logistic Regression Test Results'), {
                'fontsize': 10}, fontproperties='monospace')
        
        plt.text(0.6, 0.3, str(classification_report(y_test, y_test_preds_lr)), {
                'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!

        plt.savefig(output_pth + 'model_results.png')
        
def feature_importance_plot(cv_rfc, X_data, output_pth):
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
        importances = cv_rfc.best_estimator_.feature_importances_
        # importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        # plt.subplots_adjust(bottom=.15)
        # Save Plot
        plt.savefig(output_pth + 'features.png')

def train_models(feature_engineering_dict):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        X_train = feature_engineering_dict['X_train']
        X_test = feature_engineering_dict['X_test']
        y_train = feature_engineering_dict['y_train']
        y_test = feature_engineering_dict['y_test']

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)

        lrc.fit(X_train, y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        plt.savefig('./images/model_1.png')
        # plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig('./images/model_2.png')

        
        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')
        
        rfc_model = joblib.load('./models/rfc_model.pkl')
        lr_model = joblib.load('./models/logistic_model.pkl')

        lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
        plt.savefig('./images/model_3.png')

        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig('./images/model_4.png')
       
        
        return y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf, rfc_model, lr_model, cv_rfc
if __name__ == "__main__":
        df = import_data('./data/bank_data.csv')
        perform_eda(import_data('./data/bank_data.csv'))
        category_list=['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
        df = (encoder_helper(df,category_list))
        model_dict, X_data =  perform_feature_engineering(df)
        a,b,c,d,e,f,rfc_model, lr_model, cv_rfc = train_models(model_dict)
        # classification_report_image(a,b,c,d,e,f,'./images/')
        feature_importance_plot(cv_rfc, X_data, './')
