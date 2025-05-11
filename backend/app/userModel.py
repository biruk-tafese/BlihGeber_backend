from .db import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(20), unique=False, nullable=True)
    phone_number = db.Column(db.String(15), unique=True, nullable=False)
    user_type = db.Column(db.String(20), nullable=False, default="user")
    password = db.Column(db.String(100), nullable=False)
    

    def to_dict(self):
        return {"id": self.id, "full_name": self.full_name, "phone_number": self.phone_number}
