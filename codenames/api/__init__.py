from flask import Flask
from .routes import init_routes

app = Flask(__name__)  # pylint: disable=invalid-name
init_routes(app)
