def drift_schema(payload):
    return {
        "timestamp": payload["timestamp"],
        "dataset": payload["dataset"],
        "psi": payload["psi"],
        "drift_level": payload["drift_level"],
        "features": payload["features"]
    }

def model_health_schema(payload):
    return {
        "timestamp": payload["timestamp"],
        "model": payload["model"],
        "status": payload["status"],
        "action": payload["action"],
        "reason": payload["reason"]
    }
