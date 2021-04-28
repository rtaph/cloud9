from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# 1. Load the model (copied via the AWS CLI)
model = joblib.load("model.joblib")

# 2. Define a prediction function
def return_prediction(X):
    return model.predict(X)


# 3. Set up home page using basic html
@app.route("/")
def index():
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 
    5 climate model outputs.
    """


# 4. define a new route which will accept POST requests and return predictions
@app.route("/predict", methods=["POST"])
def rainfall_prediction():
    content = request.json
    prediction = return_prediction(np.array(content["data"]).reshape(1, -1))
    results = {"predictions": prediction.tolist()}
    return jsonify(results)