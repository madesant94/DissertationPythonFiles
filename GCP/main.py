from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from google.cloud import datastore
import pandas as pd

app = Flask(__name__)

api = Api(app)
CORS(app)  # This will enable CORS for all routes

client = datastore.Client()

DATASTORE_KEY = 'Pollution'

class DataResource(Resource):
    def get(self):
        return get_data()

    def post(self):
        return post_data()

def get_data():
    
    #try:
    #    data = pd.read_csv('sample.csv')
    #    dictionary = {}
    #    for index, row in data.iterrows():
    #        dictionary[row['code']] = round(row["predicted_pm2.5"], 2)
    #except:
    #    print('Cant find file')
    #return dictionary

    key = client.key('SingleEntity', DATASTORE_KEY)
    entity = client.get(key)

    if not entity:
        return {"error": "No data found"}, 404

    #return {"value": entity["value"]}
    return entity["value"]


def post_data():
    # Update value of the grid locally from computer with post request
    # in order to test multiple days

    data = request.get_json()

    # Handle the POST data here. 
    # if not data or not "key_name" in data or not "value" in data:
    #    return {"error": "Data format is incorrect"}, 400

    if not data or not "value" in data:
        return {"error": "Data format is incorrect"}, 400
    
    # Define a key to save values
    key = client.key('SingleEntity',DATASTORE_KEY)

    # Create a new entity using the key
    entity = datastore.Entity(key=key)

    # Update the entity's value
    entity.update({
        "value": data["value"]
    })

    # Save the entity
    client.put(entity)
    
    # return data
    return {"success": True}

api.add_resource(DataResource, '/data')

if __name__ == '__main__':
    app.run(debug=True)