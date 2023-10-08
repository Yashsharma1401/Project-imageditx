from datetime import datetime, timedelta
import os,json
from __init__ import app,db
from models import Images
with open("config.json","r") as d:
    directories=json.load(d)['directories']

def check_exceeds_10_days(given_datetime):
    current_datetime = datetime.now()
    delta = current_datetime - given_datetime
    if delta > timedelta(days=5): 
        return True
    else:
        return False

def deleteImages():
    with app.app_context():
        imgEdits=Images.query.all()
        for img in imgEdits:
            if check_exceeds_10_days(img.date):
                try:
                    path=os.path.abspath("../"+directories['savedImg']+"/"+str(img.userId)+"/"+img.image)
                    os.remove(path,dir_fd=None)
                    db.session.delete(img)
                    db.session.commit()
                except:
                    pass
            else:
                pass
        
        
        

