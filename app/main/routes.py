import os, sys
import redis
from flask import current_app
from app.config import *
from flask import request, redirect, render_template, flash, Blueprint, g
from werkzeug.utils import secure_filename
from app.main import bp
from rq import Queue, push_connection, pop_connection
from wtforms import Form, validators, TextField, FloatField, FileField
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

class ReusableForm(Form):
    emailID = TextField('Email Address', validators=[validators.required()])
    mod_start = FloatField('Moderator Start Time (s)', validators=[validators.required()])
    mod_end = FloatField('Moderator Start Time (s)', validators=[validators.required()])
    ipfile = FileField('File', validators=[validators.required()])
    
@bp.route('/', methods=['GET', 'POST'])
def index():
    form = ReusableForm(request.form)

    if request.method == 'POST':
        email = request.form['email']
        mod_start = request.form['start']
        mod_end = request.form['end']
        
        if "inputfile" not in request.files:
            flash('Please select a file')
            return render_template("upload.html", form=form)
        
        if mod_start >= mod_end:
            flash('Please enter valid times')
            return render_template("upload.html", form=form)

        if email=="":
            flash('Please enter an email address') 
            return render_template("upload.html", form=form)
        

        ipfile = request.files['inputfile']
        if ipfile:
            filename = secure_filename(ipfile.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            ipfile.save(filepath)
            flash("File {} has been uploaded successfully, please wait for email with inference details".format(filename))
            job_queue = Queue()
            job = job_queue.enqueue('app.process_files.run_SAD', args=(filepath, email, mod_start, mod_end), timeout=current_app.config['JOB_TIMEOUT'] )
            return redirect(request.url)
    return render_template("upload.html", form=form) 


