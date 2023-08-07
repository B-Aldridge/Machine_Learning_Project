from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
from flask_cors import CORS

# Define and connect
app = Flask(__name__)
CORS(app)  # Enable CORS for local readability

# Load the trained model from the joblib file
model = 'rf_model.joblib'
rf_classifier = joblib.load(model)

# Define an API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request
    data = request.get_json()

    # Load the DataFrame used during model training
    merged_df = pd.read_csv('merged_data.csv')

    # Create a DataFrame from the input data and apply one-hot encoding
    input_data = pd.DataFrame(data)
    input_data = pd.get_dummies(input_data, columns=input_data.columns)

    # Reorder the columns to match the training data's columns
    input_data = input_data.reindex(columns=merged_df.drop('Disease', axis=1).columns, fill_value=0)

    # Make predictions using the loaded model
    predictions = rf_classifier.predict(input_data)

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/symptom_checker')
def symptom_checker():
    return render_template('symptom_checker.html')

if __name__ == '__main__':
    app.run(debug=False)
