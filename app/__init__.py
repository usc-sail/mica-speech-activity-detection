from flask import Flask
from flask_mail import Mail
from flask_bootstrap import Bootstrap 
from app.main import bp as main

bootstrap = Bootstrap()
mail = Mail()

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('config.py')
    mail.init_app(app)
    bootstrap.init_app(app)
    app.register_blueprint(main)
    return app

app = create_app()
