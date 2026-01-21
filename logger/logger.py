
import json
from logger.mysql_logger import DBLogger

def log_event(event):
    db = DBLogger()
    db.insert_event(json.dumps(event))
