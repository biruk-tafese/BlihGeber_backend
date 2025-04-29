from flask import Flask
from .db import db
from .routes import routes

def create_app():
    app = Flask(__name__)

    # MySQL config (edit as needed)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:mysql@localhost:3306/crop'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = 'AAiT CropYieldPrediction Project'

    db.init_app(app)
    app.register_blueprint(routes)

    return app
