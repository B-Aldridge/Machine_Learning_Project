from flask import Flask, render_template, request
import pandas as pd
from flask_cors import CORS
import joblib

# Define and connect
app = Flask(__name__)
CORS(app)  # Enable CORS for local readability

# Load the saved model
model_filename = 'random_forest_model.pkl'
rf_model = joblib.load(model_filename)

# Get the list of encoded symptom columns
encoded_symptom_columns = ['weight','Symptom_1_acidity','Symptom_1_back_pain','Symptom_1_bladder_discomfort','Symptom_1_breathlessness','Symptom_1_burning_micturition','Symptom_1_chest_pain','Symptom_1_chills','Symptom_1_constipation','Symptom_1_continuous_sneezing','Symptom_1_cough','Symptom_1_cramps','Symptom_1_fatigue','Symptom_1_headache','Symptom_1_high_fever','Symptom_1_indigestion','Symptom_1_itching','Symptom_1_joint_pain','Symptom_1_mood_swings','Symptom_1_muscle_wasting','Symptom_1_muscle_weakness','Symptom_1_neck_pain','Symptom_1_pain_during_bowel_movements','Symptom_1_patches_in_throat','Symptom_1_pus_filled_pimples','Symptom_1_shivering','Symptom_1_skin_rash','Symptom_1_stiff_neck','Symptom_1_stomach_pain','Symptom_1_sunken_eyes','Symptom_1_vomiting','Symptom_1_weakness_in_limbs','Symptom_1_weight_gain','Symptom_1_weight_loss','Symptom_1_yellowish_skin','Symptom_2_abdominal_pain','Symptom_2_acidity','Symptom_2_anxiety','Symptom_2_blackheads','Symptom_2_bladder_discomfort','Symptom_2_blister','Symptom_2_breathlessness','Symptom_2_bruising','Symptom_2_chest_pain','Symptom_2_chills','Symptom_2_cold_hands_and_feets','Symptom_2_cough','Symptom_2_cramps','Symptom_2_dehydration','Symptom_2_dizziness','Symptom_2_fatigue','Symptom_2_foul_smell_of urine','Symptom_2_headache','Symptom_2_high_fever','Symptom_2_indigestion','Symptom_2_joint_pain','Symptom_2_knee_pain','Symptom_2_lethargy','Symptom_2_loss_of_appetite','Symptom_2_mood_swings','Symptom_2_nausea','Symptom_2_neck_pain','Symptom_2_nodal_skin_eruptions','Symptom_2_pain_during_bowel_movements','Symptom_2_pain_in_anal_region','Symptom_2_patches_in_throat','Symptom_2_pus_filled_pimples','Symptom_2_restlessness','Symptom_2_shivering','Symptom_2_skin_peeling','Symptom_2_skin_rash','Symptom_2_stiff_neck','Symptom_2_stomach_pain','Symptom_2_sunken_eyes','Symptom_2_sweating','Symptom_2_swelling_joints','Symptom_2_ulcers_on_tongue','Symptom_2_vomiting','Symptom_2_weakness_in_limbs','Symptom_2_weakness_of_one_body_side','Symptom_2_weight_gain','Symptom_2_weight_loss','Symptom_2_yellowish_skin','Symptom_3_abdominal_pain','Symptom_3_altered_sensorium','Symptom_3_anxiety','Symptom_3_blackheads','Symptom_3_blister','Symptom_3_bloody_stool','Symptom_3_blurred_and_distorted_vision','Symptom_3_breathlessness','Symptom_3_bruising','Symptom_3_burning_micturition','Symptom_3_chest_pain','Symptom_3_chills','Symptom_3_cold_hands_and_feets','Symptom_3_continuous_feel_of_urine','Symptom_3_cough','Symptom_3_dark_urine','Symptom_3_dehydration','Symptom_3_diarrhoea','Symptom_3_dischromic _patches','Symptom_3_dizziness','Symptom_3_extra_marital_contacts','Symptom_3_fatigue','Symptom_3_foul_smell_of urine','Symptom_3_headache','Symptom_3_high_fever','Symptom_3_hip_joint_pain','Symptom_3_joint_pain','Symptom_3_knee_pain','Symptom_3_lethargy','Symptom_3_loss_of_appetite','Symptom_3_loss_of_balance','Symptom_3_mood_swings','Symptom_3_movement_stiffness','Symptom_3_nausea','Symptom_3_neck_pain','Symptom_3_nodal_skin_eruptions','Symptom_3_obesity','Symptom_3_pain_in_anal_region','Symptom_3_red_sore_around_nose','Symptom_3_restlessness','Symptom_3_scurring','Symptom_3_silver_like_dusting','Symptom_3_skin_peeling','Symptom_3_spinning_movements','Symptom_3_stomach_pain','Symptom_3_sweating','Symptom_3_swelling_joints','Symptom_3_swelling_of_stomach','Symptom_3_ulcers_on_tongue','Symptom_3_vomiting','Symptom_3_watering_from_eyes','Symptom_3_weakness_of_one_body_side','Symptom_3_weight_loss','Symptom_3_yellowish_skin'
]
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    

    # Get symptom data from form
    symptoms = request.form.getlist('symptoms')
    
    # Create a dictionary to hold symptom data
    input_dict = {symptom: [True] if symptom in symptoms else [False] for symptom in encoded_symptom_columns}
    
    # Create a DataFrame with the selected symptoms
    input_data = pd.DataFrame(input_dict, columns=encoded_symptom_columns)

    # Make prediction using the loaded model
    predicted_disease = rf_model.predict(input_data)[0]

    # Get the probability distribution for all classes
    predicted_proba = rf_model.predict_proba(input_data)

    # Print the probability distribution
    print("Predicted Probability Distribution:")
    for disease, prob in zip(rf_model.classes_, predicted_proba[0]):
        print(f"{disease}: {prob:.4f}")

    # Pass the prediction to the HTML template for rendering
    return render_template('result.html', prediction=predicted_disease)

@app.route('/symptom_checker')
def symptom_checker():
    return render_template('symptom_checker.html')

if __name__ == '__main__':
    app.run(debug=False)
