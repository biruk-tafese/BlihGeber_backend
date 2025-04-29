import bcrypt

from flask import Blueprint, request, jsonify


from .userModel import User
from .db import db
import jwt
import datetime
import joblib
import pandas as pd
import os
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from flask import request


# from .__init__ import create_app
# app = create_app()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and expected feature columns
model = joblib.load(os.path.join(current_dir, 'Random_Forest_model.pkl'))
expected_columns = joblib.load(os.path.join(current_dir, 'model_features.pkl'))


# Dynamically extract crops and areas
crop_prefix = "Item_"
area_prefix = "Area_"
crops = sorted([col[len(crop_prefix):] for col in expected_columns if col.startswith(crop_prefix)])
areas = sorted([col[len(area_prefix):] for col in expected_columns if col.startswith(area_prefix)])


routes = Blueprint('routes', __name__)

"""
Decorator function to enforce JWT token authentication for protected routes.

This decorator checks for the presence of a valid JWT token in the `Authorization` 
header of the incoming request. If the token is missing, expired, or invalid, 
it returns an appropriate error response. If the token is valid, it retrieves 
the current user from the database and passes it as the first argument to the 
decorated function.

Args:
    f (function): The function to be decorated.

Returns:
    function: The decorated function with token authentication applied.

Raises:
    401 Unauthorized: If the token is missing, expired, or invalid.

Example:
    @token_required
    def protected_route(current_user):
        return jsonify({'message': 'This is a protected route'})
"""
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            bearer = request.headers['Authorization']
            token = bearer.split()[1] if ' ' in bearer else bearer

        if not token:
            return jsonify({'error': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, 'AAiT CropYieldPrediction Project', algorithms=['HS256'])
            current_user = User.query.get(data['user_id'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401

        return f(current_user, *args, **kwargs)
    return decorated


 # Register Endpoint
@routes.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()

    if not all(k in data for k in ('phone_number', 'full_name', 'password')):
        return jsonify({'error': 'Missing fields'}), 400

    if User.query.filter_by(phone_number=data['phone_number']).first():
        return jsonify({'error': 'This Account already exists'}), 409

    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())

    new_user = User(
        full_name=data['full_name'],
        phone_number=data['phone_number'],
        password=hashed_password.decode('utf-8')
    )
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User registered'}), 201


# Login Endpoint with JWT
@routes.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()

    if not all(k in data for k in ('phone_number', 'password')):
        return jsonify({'error': 'Missing phone number or password'}), 400

    user = User.query.filter_by(phone_number=data['phone_number']).first()
    if not user or not bcrypt.checkpw(data['password'].encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({'error': 'Invalid credentials'}), 401

    # Generate JWT token
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)  # Token expires in 2 hours
    }, 'AAiT CropYieldPrediction Project', algorithm='HS256')

    return jsonify({'message': 'Login successful', 'token': token, 'user_type': user.user_type}), 200

@routes.route('/crop')
def get_crops():
    return jsonify(crops)
@routes.route('/area')
def get_areas():
    return jsonify(areas)
@routes.route('/')
def index():
    return jsonify("Welcome to the Crop Yield Prediction API! Use /predict to make predictions.")

@routes.route('/predict', methods=['POST'])
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

@routes.route('/profile', methods=['GET'])
@token_required
def profile(current_user):
    return jsonify({'user': current_user.to_dict()}), 200

@routes.route('/update_profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if 'full_name' in data:
        current_user.full_name = data['full_name']

    if 'phone_number' in data:
        current_user.phone_number = data['phone_number']

    db.session.commit()

    return jsonify({'message': 'Profile updated successfully', 'user': current_user.to_dict()}), 200




@routes.route('/update_password', methods=['PUT'])
@token_required 
def password_update(current_user):
    data = request.get_json()

    if not all(k in data for k in ('old_password', 'new_password')):
        return jsonify({'error': 'Missing fields'}), 400

    if not bcrypt.checkpw(data['old_password'].encode('utf-8'), current_user.password.encode('utf-8')):
        return jsonify({'error': 'Old password is incorrect'}), 401

    current_user.password = bcrypt.hashpw(data['new_password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    db.session.commit()

    return jsonify({'message': 'Password updated successfully'}), 200

@routes.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    # Invalidate the token (this is a simple example; in a real app, you might want to store invalidated tokens)
    return jsonify({'message': 'Logged out successfully'}), 200