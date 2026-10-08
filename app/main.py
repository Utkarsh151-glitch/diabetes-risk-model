"""FastAPI service for the diabetes risk model: /health, /model-info, /predict, and the web form at /.

Input values are health data: they are never printed or logged.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from src.pipeline import FEATURES, METRICS_PATH, MODEL_PATH, ROOT

VERSION = "2.0.0"
DISCLAIMER = "Not medical advice. This is a statistical screening estimate, not a diagnosis. Talk to a doctor."
POPULATION = ("Trained on the Pima Indians Diabetes Database: 768 women aged 21 or older of Pima Indian heritage "
              "living near Phoenix, Arizona (US National Institute of Diabetes and Digestive and Kidney Diseases). "
              "Estimates for other groups are untested.")
UNITS = {
    "Pregnancies": "count",
    "Glucose": "mg/dL",
    "BloodPressure": "mmHg",
    "SkinThickness": "mm",
    "Insulin": "µU/mL",
    "BMI": "kg/m²",
    "DiabetesPedigreeFunction": "score",
    "Age": "years",
}
OPTIONAL = ["SkinThickness", "Insulin", "DiabetesPedigreeFunction"]


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Pregnancies: int = Field(..., ge=0, le=20, description="Number of times pregnant (count).")
    Glucose: float = Field(..., ge=40, le=300, description=(
        "Plasma glucose 2 hours into an oral glucose tolerance test, in mg/dL (not a fasting reading)."))
    BloodPressure: float = Field(..., ge=30, le=140, description="Diastolic blood pressure, in mmHg.")
    SkinThickness: Optional[float] = Field(None, ge=5, le=100, description=(
        "Triceps skinfold thickness, in mm. Optional: the training median is used if omitted."))
    Insulin: Optional[float] = Field(None, ge=10, le=900, description=(
        "2-hour serum insulin, in µU/mL. Optional: the training median is used if omitted."))
    BMI: float = Field(..., ge=10, le=70, description="Body mass index, in kg/m².")
    DiabetesPedigreeFunction: Optional[float] = Field(None, ge=0.05, le=2.5, description=(
        "Diabetes pedigree function, a family-history score (unitless). Optional: the training median is used "
        "if omitted."))
    Age: int = Field(..., ge=21, le=100, description="Age, in years.")


class Prediction(BaseModel):
    probability: float = Field(description="Estimated probability of diabetes, 0 to 1.")
    threshold: float = Field(description="Screening threshold chosen on training data.")
    above_threshold: bool
    result: str
    imputed: list[str] = Field(description="Optional fields that were missing and filled with the training median.")
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup.
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Diabetes Risk Model", version=VERSION, lifespan=lifespan)


def allowed_origins() -> list[str]:
    origins = [f"http://{h}:{p}" for h in ("localhost", "127.0.0.1") for p in (7860, 8000)]
    if os.environ.get("SPACE_HOST"):  # set by Hugging Face Spaces, e.g. user-diabetes-risk-model.hf.space
        origins.append(f"https://{os.environ['SPACE_HOST']}")
    origins += [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    return origins


app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(), allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"])


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": request.app.state.model is not None, "version": VERSION}


@app.get("/model-info")
def model_info(request: Request):
    m = request.app.state.metrics
    chosen = next(row for row in m["cv"] if row["name"] == m["model"]["name"])
    schema = PatientFeatures.model_json_schema()["properties"]
    medians = m["data"]["training_medians_after_zero_to_missing"]
    features = []
    for name in FEATURES:
        prop = schema[name]
        bounds = next((b for b in prop.get("anyOf", [prop]) if "minimum" in b), prop)
        features.append({"name": name, "unit": UNITS[name], "description": prop["description"],
                         "min": bounds["minimum"], "max": bounds["maximum"], "optional": name in OPTIONAL,
                         "training_median": medians[name]})
    return {
        "model": m["model"]["family"],
        "model_name": m["model"]["name"],
        "selection_reason": m["model"]["selection_reason"],
        "version": VERSION,
        "trained_on": m["trained_on"],
        "threshold": m["threshold"]["value"],
        "threshold_rule": m["threshold"]["rule"],
        "metrics": {"cv_roc_auc_mean": chosen["cv_auc_mean"], "cv_roc_auc_std": chosen["cv_auc_std"],
                    "test": m["test"]},
        "features": features,
        "training_population": POPULATION,
        "data": {"source": "OpenML dataset 37 (Pima Indians Diabetes Database)", "rows": m["data"]["rows"],
                 "sha256": m["data"]["sha256"]},
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures, request: Request) -> Prediction:
    values = features.model_dump()
    imputed = [name for name in OPTIONAL if values[name] is None]
    row = pd.DataFrame([[np.nan if values[f] is None else float(values[f]) for f in FEATURES]], columns=FEATURES)
    probability = float(request.app.state.model.predict_proba(row)[0, 1])
    threshold = float(request.app.state.metrics["threshold"]["value"])
    above = probability >= threshold
    return Prediction(
        probability=round(probability, 4),
        threshold=threshold,
        above_threshold=above,
        result="above the screening threshold" if above else "below the screening threshold",
        imputed=imputed,
        disclaimer=DISCLAIMER,
    )


# The web form, served from the same origin as the API. Mounted last so the API routes take precedence.
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
