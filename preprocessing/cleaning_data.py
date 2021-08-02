#import libraries
import pandas as pd
import numpy as np

def preprocess(json_data):
    dict_of_expected_outcome = {
        "Living area": {'type': 'int','optional':False,'default': []},
        "Bedroom": {'type': 'int', 'optional': False, 'default': []},
        "Province": {
            'type': 'str',
            'optional': True,
            'default': [
                'Brussels', 'Oost-vlaanderen', 'Vlaams-brabant', 'Luik', 'Namen',
                'Luxemburg', 'West-vlaanderen', 'Antwerpen', 'Henegouwen',
                'Waals-brabant', 'Limburg']},
        "Property Type": {'type': 'str', 'optional': True, 'default': ['Apartment','House']},
        "Property Subtype": {
            'type': 'str',
            'optional': True,
            'default': [
                'Apartment', 'Town-house', 'House', 'Villa', 'Penthouse',
                'Mansion', 'Studio', 'Exceptional property', 'Kot', 'Duplex',
                'Triplex', 'Ground floor', 'Bungalow', 'Loft', 'Chalet',
                'Service flat', 'Castle', 'Farmhouse', 'Country house',
                'Manor house', 'Other properties']
            },
        "Surface of the plot": {'type': 'int', 'optional': True, 'default': []},
        "HasGarden": {'type': 'str', 'optional': True, 'default': ['Yes','No']},
        "Garden surface": {'type': 'int', 'optional': True, 'default': []},
        "Kitchen Type": {
            'type': 'str',
            'optional': True,
            'default': ['Equipped', 'Semi-equipped', 'Not installed']},
        "Swimming pool": {'type': 'str', 'optional': True, 'default': ['Yes','No']},
        "Furnished": {'type': 'str', 'optional': True, 'default': ['Yes','No']},
        "HasFireplace": {'type': 'str', 'optional': True, 'default': ['Yes','No']},
        "HasTerrace": {'type': 'str', 'optional': True, 'default': ['Yes','No']},
        "Terrace surface": {'type': 'int', 'optional': True, 'default': []},
        "Number of frontages": {'type': 'int', 'optional': True, 'default': []},
        "Building condition": {'type': 'str', 'optional': True, 'default': ['As new','Good','To renovate']}
    }

    #Check if non-optional features are provided
    for feature in dict_of_expected_outcome.keys():
        if not dict_of_expected_outcome[feature]['optional']:
            if feature not in json_data.keys():
                raise ValueError(f'The feature {feature} is missing')

    #Check if the provided features match the accepted columns
    new_json_data = json_data.copy()
    for feature, value in json_data.items():
        if feature not in dict_of_expected_outcome.keys():
            raise ValueError(f'{feature} is not a valid entry')
        if type(value) != dict_of_expected_outcome[feature]['type']:
            raise ValueError(f'{feature}:{value} of {type(value)} should be {dict_of_expected_outcome[feature]["type"]}')
        if dict_of_expected_outcome[feature]['type'] == str and len(dict_of_expected_outcome[feature]['default'])>0:
            if value not in dict_of_expected_outcome[feature]['default']:
                raise ValueError(f'Possible entries for {feature} are {dict_of_expected_outcome[feature]["default"]}')
            else:
                #convert data
                new_json_data[f"{feature}_{value}"] = 1
                del new_json_data[feature]

    return [new_json_data]
