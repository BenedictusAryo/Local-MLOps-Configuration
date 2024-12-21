"""
Pydantic Object Models for API
"""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    Fresh: float
    Milk: float
    Grocery: float
    Frozen: float
    Detergents_Paper: float
    Delicassen: float
    Channel: int


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
