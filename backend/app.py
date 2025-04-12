import flask
from flask import request, jsonify
import joblib
import pandas as pd
# Load model and expected feature columns
model = joblib.load('../notebook/Random_Forest_model.pkl')
expected_columns = joblib.load('../notebook/model_features.pkl')

# Dynamically extract crops and areas
crop_prefix = "Item_"
area_prefix = "Area_"
crops = sorted([col[len(crop_prefix):] for col in expected_columns if col.startswith(crop_prefix)])
areas = sorted([col[len(area_prefix):] for col in expected_columns if col.startswith(area_prefix)])

app = flask.Flask(__name__)

@app.route('/crop')
def get_crops():
    return jsonify(crops)
@app.route('/area')
def get_areas():
    return jsonify(areas)
@app.route('/')
def index():
    return "Welcome to the Crop Yield Prediction API! Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract input values
    avg_rain = data.get('average_rain_fall_mm_per_year', 0.0)
    pesticides = data.get('pesticides_tonnes', 0.0)
    avg_temp = data.get('avg_temp', 0.0)
    selected_crop = data.get('crop', '')
    selected_area = data.get('area', '')

    # Build input dict
    input_data = {
        'average_rain_fall_mm_per_year': [avg_rain],
        'pesticides_tonnes': [pesticides],
        'avg_temp': [avg_temp],
    }

    # One-hot encode crop and area
    for crop in crops:
        input_data[f'Item_{crop}'] = [1 if crop == selected_crop else 0]
    for area in areas:
        input_data[f'Area_{area}'] = [1 if area == selected_area else 0]

    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)

    # Reindex to match expected columns (fill missing with 0)
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df)
    return jsonify({'predicted_yield': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)