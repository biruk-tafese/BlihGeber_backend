import streamlit as st
import pandas as pd
import joblib

## Load model and expected feature columns
model = joblib.load('../notebook/Random_Forest_model.pkl')  # Adjust path if necessary
expected_columns = joblib.load('../notebook/model_features.pkl')  # Adjust path if necessary

# Dynamically extract crops and areas
crop_prefix = "Item_"
area_prefix = "Area_"
crops = sorted([col[len(crop_prefix):] for col in expected_columns if col.startswith(crop_prefix)])
areas = sorted([col[len(area_prefix):] for col in expected_columns if col.startswith(area_prefix)])

st.title("🌾 Crop Yield Prediction App")

# Input fields
avg_rain = st.number_input("Average Rainfall (mm/year)", min_value=0.0, value=1200.0)
pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, value=500.0)
avg_temp = st.number_input("Average Temperature (°C)", min_value=-10.0, value=18.5)

# Dropdowns (dynamically loaded)
selected_crop = st.selectbox("Select Crop", crops)
selected_area = st.selectbox("Select Country", areas)

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

# Prediction
if st.button("Predict Yield"):
    print(input_df)
    prediction = model.predict(input_df)
    st.success(f"🌱 Predicted Yield: {prediction[0]:.2f}")
