# A Flask-API for a Prediction Model: Belgium Real Estate Prices

This repository showcases a Flask application that creates predictions from a scikit-learn model for the real estates in Belgium. Once the Flask app is run, the presaved model returns the predicted price based on given features. If the model is not found, it will first train the given data, save it as a model, and then give a prediction. 

The model is deployed to Heroku which then will be used by web-developers to embed it in their websites.


### Project Division: 
3 main components of this project:
- model.py : This file contains a class that handles cleaning the initial dataset, training a machine learning model, and saving it with joblib.
- cleaning_data.py : This file preprocess the user input, makes sure that the user give the correct data type and filled in the required features in order to predict the price.
- app.py : This file has the Flask API, once its run, it will receive the user input as a json file, convert it to dataframe, and fit it to the presaved model and return a prediction price. GET methods that were defined as a route will return the expected inputs and data types from the user.

#### Dependencies
- Flask
- Scikit-learn
- Pandas
- Numpy  

```
pip install -r requirements.txt
```

#### Running API
To run the app, open your terminal in the same directory as your app.py script and run this command.  
```
pip install -r requirements.txt
```

### Endpoints
**1. / (GET)** <br>
Returns a text : `Alive!!`

**2. /predict (GET)** <br>
Returns an explanation of expected column names and valid enteries for prediction to work. Here's a sample input: <br>
```
{
        "Living area": {'type': int,'optional':False,'default': []},
        "Bedroom": {'type': int, 'optional': False, 'default': []}, 
        "Province": {
            'type': str,
            'optional': False,
            'default': [
                'Brussels', 'Oost-vlaanderen', 'Vlaams-brabant', 'Luik', 'Namen',
                'Luxemburg', 'West-vlaanderen', 'Antwerpen', 'Henegouwen',
                'Waals-brabant', 'Limburg']}, 
        "Property Type": {'type': str, 'optional': False, 'default': ['Apartment','House']},
        "Property Subtype": {
            'type': str,
            'optional': True,
            'default': [
                'Apartment', 'Town-house', 'House', 'Villa', 'Penthouse',
                'Mansion', 'Studio', 'Exceptional property', 'Kot', 'Duplex',
                'Triplex', 'Ground floor', 'Bungalow', 'Loft', 'Chalet',
                'Service flat', 'Castle', 'Farmhouse', 'Country house',
                'Manor house', 'Other properties']
            },
        "Surface of the plot": {'type': int, 'optional': True, 'default': []},
        "HasGarden": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "Garden surface": {'type': int, 'optional': True, 'default': []},
        "Kitchen Type": {
            'type': str,
            'optional': True,
            'default': ['Equipped', 'Semi-equipped', 'Not installed']},
        "Swimming pool": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "Furnished": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "HasFireplace": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "HasTerrace": {'type': str, 'optional': True, 'default': ['Yes','No']},
        "Terrace surface": {'type': int, 'optional': True, 'default': []},
        "Number of frontages": {'type': int, 'optional': True, 'default': []},
        "Building condition": {'type': str, 'optional': True, 'default': ['As new','Good','To renovate']}
}
```


**3. /predict (POST)** <br>
Returns an array of prediction based on given JSON objects representing the independent variables. Here's a sample input: <br>
``` 
[

{ "Living area": 100, "Bedroom" :3, "Province" : "Brussels", "Property Type" :"Apartment", "HasGarden":"No"  }

] 

```
Sample output:
```
{ "Prediction Price": [ 326.41892486583856 ] }
```




