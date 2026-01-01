# aegis/api/schemas.py

from pydantic import BaseModel
from typing import List


class ExplainRequest(BaseModel):
    dataset: str
    index: int


class FeatureImpact(BaseModel):
    feature: str
    impact: float
    direction: str


class ExplainResponse(BaseModel):
    alert_id: str
    prediction: str
    confidence: float
    top_contributors: List[FeatureImpact]
    summary: str
