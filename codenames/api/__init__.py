from flask import Flask
from .routes import init_routes


def create_app() -> Flask:
    app = Flask(__name__)
    init_routes(app)
    return app
