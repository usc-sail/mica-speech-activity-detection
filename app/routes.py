import os, sys
from app import app
from app.config import *
from flask import request, redirect, render_template, flash
from werkzeug.utils import secure_filename
sys.path.insert(0, '/Users/rabbeh/Projects/ITU/app')
from process_files import run_VAD
from app.worker import conn
from rq import Queue


#def run_VAD(data_filepath):
#    if not os.path.exists(OUT_DIR):
#        os.mkdir(OUT_DIR)
#    
#    filename = data_filepath.rsplit('/')[-1].split('.')[0]
#    out_dir = os.path.join(OUT_DIR, filename)
#    if not os.path.exists(out_dir):
#        os.mkdir(out_dir)
#    paths_file = os.path.join(out_dir, 'file_list.txt')
#
#    fw = open(paths_file, 'w')
#    fw.write('{}\n'.format(data_filepath))
#    fw.close()
#
#    os.system('bash {}/perform_SAD.sh {} {}'.format(SCRIPT_DIR, paths_file, out_dir))

ALLOWED_EXTENSIONS = set(['mp4', 'mkv'])
job_queue = Queue(connection=conn)
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            flash("File {} has been uploaded successfully, please wait for email with inference details".format(filename))
            msg = "File {} has been processed".format(filename)
            #send_email(subject=MAIL_SUBJECT, sender=MAIL_USERNAME, recipients=MAIL_RECIPIENTS, text_body=msg, html_body="")
            job = job_queue.enqueue(run_VAD, filepath)
#            run_VAD(UPLOAD_FOLDER, '/Users/rabbeh/Projects/temp/')
            return redirect(request.url)
    return render_template("upload.html") 


