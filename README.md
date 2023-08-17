# Disease Prediction Model
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

# Goal

This project revolves around creating a machine learning model to predict diseases based on a range of symptoms. We've integrated this predictive model into an interactive user interface using Flask.

# Getting Started

Our project began with data exploration and defining its scope. We decided to work with a dataset from [Kaggle](https://kaggle.com) and imported it into a Jupyter Notebook. This marked the start of our data preprocessing journey. We handled several CSV files containing disease and severity information. By merging them based on common columns and discarding irrelevant ones, we prepared the data for model training. Throughout the preprocessing, we encountered different challenges, which we overcame by focusing solely on diseases and symptoms, aligning with our initial goal.

# Model Training

The model was trained using the Random Forest algorithm, chosen based on testing results and project requirements. Given the medical dataset's mixture of symptoms and diseases, Random Forest proved fitting for disease prediction. The model achieved an accuracy score of **0.9522357723577236** or **95.22%**, highlighting its strong predictive ability.

Model was saved under **random_forest_model.pkl**

# Flask Intergration
The Flask web framework is used to create an interactive user interface for our model. Here's how it works:

1. The main page (`index.html`) presents users with a dashboard offering a choice between the "Symptom Checker" and "Schedule a Doctor Visit" (via [Zocdoc](https://www.zocdoc.com/)).
2. Upon going to the "Symptom Checker" link, filling out the form, the Flask app processes the input and makes a prediction.
3. The resulting disease is displayed on the result page (`result.html`).

## Usage

1. Clone this repository to your local machine.
2. Install the required Python libraries listed if not already installed.
3. Run the Flask app using `python app.py`.
4. Open a web browser and navigate to `http://localhost:5000/symptom_checker`.


# Credits
Team responsible for the formation of this project:
|Member|Github|
|------|------|
Andrew Lounsbury| https://github.com/a676-code
Kirill Zavalin| https://github.com/KZavalin
Joshua Aldridge| https://github.com/B-Aldridge
Madi Subaiti| https://github.com/mssubaiti
Seth Beverley| https://github.com/SNbeverley

# Resources

[Kaggle](https://kaggle.com)
