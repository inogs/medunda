from datetime import datetime

def date_from_str(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")