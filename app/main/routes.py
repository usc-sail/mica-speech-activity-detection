import os, sys
import redis
from flask import current_app
from app.config import *
from flask import request, redirect, render_template, flash, Blueprint, g
from werkzeug.utils import secure_filename
from app.main import bp
from rq import Queue, push_connection, pop_connection

ALLOWED_EXTENSIONS = set(['mp4', 'mkv'])


def get_redis_connection():
    redis_connection = getattr(g, '_redis_connection', None)
    if redis_connection is None:
        redis_url = current_app.config['REDIS_URL']
        redis_connection = g._redis_connection = redis.from_url(redis_url)
    return redis_connection


@bp.before_request
def push_rq_connection():
    push_connection(get_redis_connection())


@bp.teardown_request
def pop_rq_connection(exception=None):
    pop_connection()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('exception 1')
            flash('Please select a file')
            return redirect(request.url)
        file = request.files['file']
        if not allowed_file(file.filename):
            print('exception 2')
            flash('Only .mp4/.mkv files allowed, please try again.')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash("File {} has been uploaded successfully, please wait for email with inference details".format(filename))
            job_queue = Queue()
            job = job_queue.enqueue('app.process_files.run_SAD', args=(filepath,), timeout=current_app.config['JOB_TIMEOUT'] )
            return redirect(request.url)
    return render_template("upload.html") 


