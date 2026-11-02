"""REST prediction service (FastAPI).

Run:   uvicorn serving.api:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data_generation import AIRPORTS, CARRIERS, NEW_CARRIER
from src.predict import champion_info, predict

app = FastAPI(title="Flight Delay Prediction Service", version="1.0",
              description="Champion model from the MLflow registry + online feature store")

VALID_CARRIERS = set(CARRIERS) | {NEW_CARRIER[0]}


class FlightRequest(BaseModel):
    carrier: str = Field(..., examples=["6E"])
    origin: str = Field(..., examples=["DEL"])
    dest: str = Field(..., examples=["BOM"])
    departure: str = Field(..., examples=["2025-10-01T18:30:00"])
    weather_index: float = Field(0.5, ge=0, le=5)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return champion_info()


@app.post("/predict")
def predict_endpoint(req: FlightRequest):
    if req.origin not in AIRPORTS or req.dest not in AIRPORTS or req.origin == req.dest:
        raise HTTPException(422, "Unknown or identical origin/destination airport")
    if req.carrier not in VALID_CARRIERS:
        raise HTTPException(422, f"Unknown carrier; valid: {sorted(VALID_CARRIERS)}")
    return predict(req.carrier, req.origin, req.dest, pd.Timestamp(req.departure), req.weather_index)
