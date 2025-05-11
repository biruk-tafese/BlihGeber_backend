import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the datasets
yield_data = pd.read_csv('data/yield.csv')
pesticides_data = pd.read_csv('data/pesticides.csv')
rainfall_data = pd.read_csv('data/rainfall.csv')
temp_data = pd.read_csv('data/temp.csv')

# Merge datasets if necessary (this depends on your data structure)
# Assuming all datasets have a common key to merge on, e.g., 'Year'
data = yield_data.merge(pesticides_data, on='Year').merge(rainfall_data, on='Year').merge(temp_data, on='Year')

# Sample a smaller subset of the data to reduce memory usage
data_sample = data.sample(frac=0.1, random_state=42)  # Use 10% of the data

# Prepare features and target variable
X = data_sample.drop(columns=['yield'])  # Adjust this based on your actual column names
y = data_sample['yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model with fewer trees to reduce memory usage
model = RandomForestRegressor(n_estimators=50)  # Reduce the number of trees
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
joblib.dump(model, 'notebook/Random_Forest_model.pkl')

# Save the expected feature columns to a .pkl file
joblib.dump(X.columns.tolist(), 'notebook/model_features.pkl')

print("Model and features saved successfully.")
