import os
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from model.model import Model
from preprocessing.cleaning_data import preprocess

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "Alive!!"

# Post request that receives the data of a house in JSON format and returns the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if regressor:
        try:
            json_ = request.json  # Take all available entries
            json_ = preprocess(json_[0]) #preprocess the json file
            query = pd.DataFrame(json_)
            query = query.reindex(columns=model_columns, fill_value=0) #Convert column names to model's column names
            prediction = list(regressor.predict(query)) #Prediction
            return jsonify({'prediction': list(prediction)}) #Return the result of prediction

        except Exception as e:
            return jsonify({'Error': str(e), "trace": traceback.format_exc()})

        except ValueError as err:
            return jsonify({'Error': str(err.args), "trace": traceback.format_exc()})

    else:
        print("Train the dataset first")
        return "No model found here!"


# Get request returning the explanation of expected data and format
@app.route('/predict', methods=['GET'])
def str_format():

    dict_of_expected_outcome = {
        "Living area": {'type': int,'optional':False,'default': []},
        "Property Type": {'type': str, 'optional': False, 'default': ['Apartment','House']},
        "Property Subtype": {
            'type': str,
            'optional': False,
            'default': [
                'Apartment', 'Town-house', 'House', 'Villa', 'Penthouse',
                'Mansion', 'Studio', 'Exceptional property', 'Kot', 'Duplex',
                'Triplex', 'Ground floor', 'Bungalow', 'Loft', 'Chalet',
                'Service flat', 'Castle', 'Farmhouse', 'Country house',
                'Manor house', 'Other properties']
            },
        "Bedroom": {'type': int, 'optional': False, 'default': []},
        "Province": {'type': str, 'optional': False, 'default': ['Brussels']},
        "Surface of the plot": {'type': int, 'optional': True, 'default': []},
        "HasGarden": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "Garden surface": {'type': int, 'optional': True, 'default': []},
        "Kitchen Type": {
            'type': str,
            'optional': True,
            'default': ['Equipped', 'Semi-equipped', 'Not installed', 'Unknown']},
        "Swimming pool": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "Furnished": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "HasFireplace": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "HasTerrace": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "Terrace surface": {'type': int, 'optional': True, 'default': []},
        "Number of frontages": {'type': int, 'optional': True, 'default': []},
        "Building condition": {'type': str, 'optional': True, 'default': ["As new","Good","To renovate"]}
    }
    ...
    for key, value in dict_of_expected_outcome.items():
        print(f"{key}: {value}")

    return (f"{key}: {value}")


if __name__ == '__main__':
    try:
        model_directory = '/Users/cerenmorey/PycharmProjects/API-Deployment/model/model.pkl'
        model = Model(model_directory)
        model_file_path = model.model_path
        model_columns_file_path = model.model_column_path

        regressor = joblib.load(model_file_path)
        print("Model loaded")
        model_columns = joblib.load(model_columns_file_path)
        print("Model columns loaded")

    except Exception as e:
        print("No model found here! Need to train first.")
        print(str(e))
        regressor = None
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", threaded=True, port=port)