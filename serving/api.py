"""REST prediction service (FastAPI), hardened for exposure.

Run:   uvicorn serving.api:app --port 8000
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.data_generation import AIRPORTS, CARRIERS, NEW_CARRIER
from src.predict import champion_info, predict

Airport = Literal[tuple(AIRPORTS)]                                  # type: ignore[valid-type]
Carrier = Literal[tuple(list(CARRIERS) + [NEW_CARRIER[0]])]         # type: ignore[valid-type]

app = FastAPI(title="Flight Delay Prediction Service", version="1.1",
              description="Champion model from the MLflow registry + online feature store")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver", "*.local"])


class FlightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")           # reject unexpected fields
    carrier: Carrier = Field(..., examples=["6E"])
    origin: Airport = Field(..., examples=["DEL"])
    dest: Airport = Field(..., examples=["BOM"])
    departure: datetime = Field(..., examples=["2025-10-01T18:30:00"])
    weather_index: float = Field(0.5, ge=0, le=5)

    @model_validator(mode="after")
    def _different_airports(self):
        if self.origin == self.dest:
            raise ValueError("origin and dest must differ")
        if not (2020 <= self.departure.year <= 2035):
            raise ValueError("departure year out of supported range")
        return self


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        return champion_info()
    except Exception:
        raise HTTPException(503, "No champion model available — run `python -m src.pipeline` first")


@app.post("/predict")
def predict_endpoint(req: FlightRequest):
    try:
        dep = pd.Timestamp(req.departure).tz_localize(None) if req.departure.tzinfo else pd.Timestamp(req.departure)
        return predict(req.carrier, req.origin, req.dest, dep, req.weather_index)
    except HTTPException:
        raise
    except Exception:
        # never leak internals (paths, stack traces) to API clients
        raise HTTPException(503, "Prediction service unavailable")
