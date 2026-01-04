from aegis.mlops.mongo_client import drift_col
from datetime import datetime

drift_col.insert_one({
    "timestamp": datetime.utcnow(),
    "dataset": "UNSW",
    "psi": 0.42,
    "drift_level": "medium",
    "features": ["src_bytes", "dst_bytes"]
})

print("Inserted")
