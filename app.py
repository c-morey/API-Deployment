import sys
import os
import shutil
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
app.config["DEBUG"] = True

model_directory = '/Users/cerenmorey/PycharmProjects/API-Deployment/model'
model_file_name = '%s/model.pkl' % model_directory
model_collumns_file_name = '%s/model_columns.pkl' % model_directory


# Create some test data for our catalog in the form of a list of dictionaries.
@app.route('/', methods=['GET'])
def home():
    return "Alive"


# Post request that receives the data of a house in JSON format and returns the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if regressor:
        try:
            data = request.json  # Take all available entries
            data_df = pd.get_dummies(pd.DataFrame(data)) #Create a df from it with dummy variables for categorical data

            data_df = data_df.reindex(columns=model_columns, fill_value=0) #Convert column names to model's column names
            data_df.get("Living area", None)
            data_df.get("Bedroom", None)

            prediction = list(regressor.predict(data_df)) #Prediction
            return jsonify({'prediction': list(prediction)}) #Return the result of prediction

        except Exception as e:
            return jsonify({'Error': str(e), "trace": traceback.format_exc()})

    else:
        print("Train the dataset first")
        return "No model found here!"


# Get request returning the explanation of expected data and format
@app.route('/predict', methods=['GET'])
def str_format():

    explanation = "Expected input and their format: " \
                  "{'area': int , " \
                  "'property-type': 'Apartment' | 'House'" \
                  " 'zip-code': Optional[int]" \
                  " 'Province': ['Brussels' | 'Oost-vlaanderen' | 'Vlaams-brabant' | 'Luik' | 'Namen' | 'Luxemburg' | 'West-vlaanderen' | 'Antwerpen'| 'Henegouwen' | 'Waals-brabant' | 'Limburg']"\
                  "'rooms-number': Optional[int]" \
                  "'garden': Optional['Yes' | 'No'] " \
                  " 'garden-area': Optional[int]," \
                  " 'swimming-pool': Optional['Yes' | 'No']," \
                  " 'furnished': 'Optional['Yes' | 'No']'," \
                  " 'open-fire': Optional['Yes' | 'No'], " \
                  " 'terrace': Optional['Yes' | 'No']," \
                  " 'terrace-area': Optional[int]," \
                  " 'garden': Optional['Yes','No'], " \
                  " 'garden-area: Optional[int]," \
                  "'facade-number': Optional[int]" \
                  " 'building-condition' : Optional['As New'|'Good'|'Just renovated'|'To be done up'|'To renovate'|'To restore']"
    return explanation


if __name__ == '__main__':
    try:
        regressor = joblib.load(model_file_name)
        print("Model loaded")
        model_columns = joblib.load(model_collumns_file_name)
        print("Model columns loaded")
    except Exception as e:
        print("No model found here! Need to train first.")
        print(str(e))
        regressor = None
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", threaded=True, port=port)
