from preprocessing.cleaning_data import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def predict():
    #Call the preprocessing function
    df = preprocessing()
    df = df.select_dtypes(exclude=['object'])

    # #Define the dependent and independent variables
    X = df.drop(['Price','Location'], axis=1)
    y = df['Price']
    print(X)
    print(y)

    #Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #Create the regression model
    regressor = LinearRegression()
    #Fit the model to the train set
    regressor.fit(X_train, y_train)
    #Get the score of the model and the prediction
    regressor.score(X_train, y_train)
    y_predict = regressor.predict(X_test)

    print("Accuracy: {}%".format(round(regressor.score(X_test, y_test), 2)))

    #Saved the model
    joblib.dump(regressor,'model.pkl')

    #Load the regression model that was just saved
    regressor = joblib.load('model.pkl')
    print("Model loaded")

    # Saving the data columns from training
    model_columns = list(X.columns)
    joblib.dump(model_columns, 'model_columns.pkl')
    print("Models columns dumped!", model_columns)

result = predict()
result
