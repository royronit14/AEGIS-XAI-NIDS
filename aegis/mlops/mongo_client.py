from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["aegis_db"]

drift_col = db["drift_history"]
health_col = db["model_health"]
alert_col = db["alerts"]
