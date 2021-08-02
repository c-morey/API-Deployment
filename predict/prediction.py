from preprocessing.cleaning_data import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

import joblib


def predict():
    #call the preprocessing function
    df = preprocessing()

    #feature scaling
    X = df.drop('Price', axis=1)
    y = df['Price']

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

    #standard scaling our data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #check their shape
    X_train.shape, X_test.shape

    #model building and evaluation
    #Polynomial regression
    polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)

    #convert X data
    poly_features_train = polynomial_converter.fit_transform(X_train)
    poly_features_test = polynomial_converter.fit_transform(X_test)

    #elastic model and fitting
    elastic_model = ElasticNetCV(l1_ratio=1, tol=0.01)
    elastic_model.fit(poly_features_train, y_train)
    y_pred_poly = elastic_model.predict(poly_features_test)
    #testing model

    poly_mae = mean_absolute_error(y_test, y_pred_poly)
    poly_r2_score = r2_score(y_test, y_pred_poly)
    print("MAE for Polynomial: ", poly_mae)
    print('R2 SCORE for Polynomial: ', poly_r2_score)

    #check for minimum predicted value
    y_pred_poly.min()

    #plot the regression
    plt.figure(figsize=(10, 8))
    sns.regplot(y_pred_poly, y_test)

    # save the model
    joblib.dump(elastic_model, 'model.pkl')

    # Load the model that was just saved
    model = joblib.load('model.pkl')
    print("Model loaded")

    # Saving the data columns from training
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("Models columns dumped!", model_columns)

result = predict()
result




