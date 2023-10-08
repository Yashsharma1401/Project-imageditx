
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

app=Flask(__name__,template_folder="Templates")
CORS(app)
app.secret_key='sdx2323@3343zbhcfew3rr3343@@###$2ffr454'
# app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

from config import Config
app.config.from_object(Config)
db=SQLAlchemy(app)   

from views import *
from helper import deleteImages

with app.app_context():
    db.create_all()


    
# scheduler = BackgroundScheduler(daemon=True)
# scheduler.add_job(deleteImages, 'cron', hour=0, minute=0)
# scheduler.start()       

if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=4000)