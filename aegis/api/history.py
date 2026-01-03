from fastapi import APIRouter
from aegis.mlops.mongo_client import drift_col, health_col, alert_col

router = APIRouter(prefix="/history", tags=["History"])


@router.get("/drift")
def get_drift_history(limit: int = 20):
    data = list(
        drift_col.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return {"count": len(data), "data": data}


@router.get("/performance")
def get_performance_history(limit: int = 20):
    data = list(
        health_col.find({}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return {"count": len(data), "data": data}


@router.get("/retraining")
def get_retraining_history(limit: int = 20):
    data = list(
        alert_col.find(
            {"state": "retrain_scheduled"}, {"_id": 0}
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    return {"count": len(data), "data": data}
