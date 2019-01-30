from flask import Flask
from flask_mail import Mail
from redis import Redis
import rq

app = Flask(__name__)
app.config.from_pyfile('config.py')
mail = Mail(app)
mail.init_app(app)
app.redis = Redis.from_url(app.config['REDIS_URL'])
app.task_queue = rq.Queue('default', connection=app.redis)

from app import routes
