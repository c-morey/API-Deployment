#import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

class Model():
    def __init__(self, model_directory):
        self.df = pd.read_csv('/Users/cerenmorey/Desktop/BeCode/becode_projects/immolisa_scrapped_data - final_vers.csv')
        self.model_path = Path(model_directory)
        if self.model_path.is_file():
            self.model_column_path = self.model_path.parent.joinpath("model_columns.pkl")
        else:
            print("No models found, training new model.")
            self.model, self.model_columns = self.train_model()
            self.model_path = self.save_model()
            self.model_column_path = self.save_model_columns()

    def clean_df(self):
        df = self.df
        # drop duplicates
        df.drop_duplicates(['Location', 'Property Type', 'Property Subtype', 'Price', 'Bedroom', 'Living area'],
                           keep='first', inplace=True, ignore_index=True)

        # clean the price column
        df['Price'] = df['Price'].str.replace('.', '')
        df['Price'] = df['Price'].str.replace(',', '')
        df['Price'].replace({'No': 275000}, inplace=True)  # replace the wrong entry with a correct one
        df = df[(df['Price'] != 'Make') & (df['Price'] != 'Reserve')]  # remove texts
        df['Price'] = df['Price'].astype(int)  # set the dtype to int

        # remove mixed-use building and apartment-blocks from subtypes
        df = df[~df['Property Subtype'].isin(['Mixed-use building'])]
        df = df[~df['Property Subtype'].isin(['Apartment block'])]

        # fill-in the missing values in Living area with median
        df['Living area'].median()
        df['Living area'].replace({0: 132}, inplace=True)

        # check unique entries in each categorical columns and do necessary cleaning if needed
        df['Property Type'].unique()
        df['Property Subtype'].unique()
        df['Province'].unique()
        df['Kitchen Type'].unique()
        # reorganize kitchen types to reduce number of diff categories
        df['Kitchen Type'] = df['Kitchen Type'].replace(
            {"USA hyper equipped": "Equipped", "Installed": "Equipped", "Hyper equipped": "Equipped",
             "USA installed": "Equipped",
             "USA semi equipped": "Semi-equipped", "Semi equipped": "Semi-equipped",
             "USA uninstalled": "Not installed", "0": "Unknown"})
        df['Furnished'].unique()
        # replace 0 - which is a missing value - with 'Unknown'
        df['Furnished'].replace({'0': 'Unknown'}, inplace=True)

        df['HasFireplace'].unique()
        df['HasGarden'].unique()
        df['HasTerrace'].unique()
        df['Swimming pool'].unique()
        df['Building condition'].unique()
        # reorganize the building condition to reduce number of diff categories
        df['Building condition'] = df['Building condition'].replace(
            {"Just renovated": "As new", "To be done up": "To renovate",
             "To restore": "To renovate"})
        df['Number of frontages'].unique()
        df['How many fireplaces?'].unique()

        df.corr()["Price"].sort_values(ascending=False)
        # 'Location' column gives a negative corr. Because we have the province information, I will remove the location column
        df = df.drop('Location', axis=1)

        # check again the correlation of features with Price
        df.corr()["Price"].sort_values(ascending=False)

        # log transformation of highly skewed columns
        df['Price'] = np.log(df['Price'])
        df['Living area'] = np.log(df['Living area'])
        df['Surface of the plot'] = np.log(
            (df['Surface of the plot'] + 1))  # because this column has 0 as a value, we apply log(x+1) transformation

        # check object and numeric datatypes and their shape
        df_object = df.select_dtypes(include="object")
        df_numeric = df.select_dtypes(exclude="object")

        # create dummy variables for object types and combine it with the dataframe
        df = pd.get_dummies(df, columns=df_object.columns)
        return df

    def train_model(self):
        self.df = self.clean_df()
        df = self.df

        # feature scaling
        X = df.drop('Price', axis=1)
        y = df['Price']

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

        # standard scaling our data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # check their shape
        X_train.shape, X_test.shape

        # model building and evaluation
        # Polynomial regression
        polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)

        # convert X data
        poly_features_train = polynomial_converter.fit_transform(X_train)
        poly_features_test = polynomial_converter.fit_transform(X_test)

        # elastic model and fitting
        elastic_model = ElasticNetCV(l1_ratio=1, tol=0.01)
        return elastic_model.fit(poly_features_train, y_train), list(X.columns)

        # Testing model performance
        # y_pred_poly = elastic_model.predict(poly_features_test)
        # poly_mae = mean_absolute_error(y_test, y_pred_poly)
        # poly_r2_score = r2_score(y_test, y_pred_poly)
        # print("MAE for Polynomial: ", poly_mae)
        # print('R2 SCORE for Polynomial: ', poly_r2_score)

    def save_model(self):
        # save the model
        model_path = joblib.dump(self.model, 'model/model.pkl')[0]
        print("Model saved")
        return model_path

    def save_model_columns(self):
        # Saving the data columns from training
        model_columns_path = joblib.dump(self.model_columns, 'model/model_columns.pkl')[0]
        print("Models columns dumped!", self.model_columns)
        return model_columns_path

# model_directory = '/Users/cerenmorey/PycharmProjects/API-Deployment/model/model.pkl'
# model = Model(model_directory)
# model_file_name = model.model_path.name
# model_columns_file_name = model.model_column_path.name

