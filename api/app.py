from flask import Flask
from flask_cors import CORS

from api.blueprint import app_blueprint

app = Flask(__name__)

app.register_blueprint(app_blueprint)

CORS(app)
