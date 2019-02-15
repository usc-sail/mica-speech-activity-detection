import sys, os
from flask import Flask
from flask import current_app
from app.config import *
from app.email import send_email

OUT_DIR='/Users/rabbeh/Projects/ITU/flask_proj/mod_proj/app/out_dir'
SCRIPT_DIR='/Users/rabbeh/Projects/VAD/mica-speech-activity-detection'
def run_SAD(data_filepath, emailID, mod_start, mod_end):
    if emailID != "":
        MAIL_RECIPIENTS=[emailID]
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    
    filename = data_filepath.rsplit('/')[-1].split('.')[0]
    out_dir = os.path.join(OUT_DIR, filename)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    paths_file = os.path.join(out_dir, 'file_list.txt')

    fw = open(paths_file, 'w')
    fw.write('{}\n'.format(data_filepath))
    fw.close()

    os.system('bash {}/perform_SAD.sh {} {}'.format(SCRIPT_DIR, paths_file, out_dir))
    msg = f'File {filename} has been processed'
    #msg = Message('File {} has been processed'.format(filename), sender=MAIL_USERNAME, recipients = MAIL_RECIPIENTS)
    #mail.send(msg)   
    send_email(subject = MAIL_SUBJECT, sender = MAIL_USERNAME, recipients = MAIL_RECIPIENTS, text_body=msg, html_body='', sync=True)

if __name__=='__main__':
    data_path = sys.argv[1]
    run_SAD(data_path)
