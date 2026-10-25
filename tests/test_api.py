import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.pipeline import METRICS_PATH

FULL = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 25, "Insulin": 100,
        "BMI": 28.5, "DiabetesPedigreeFunction": 0.4, "Age": 35}
REQUIRED_ONLY = {k: v for k, v in FULL.items() if k not in ("SkinThickness", "Insulin", "DiabetesPedigreeFunction")}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_valid_input_returns_probability(client):
    r = client.post("/predict", json=FULL)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["imputed"] == []
    assert body["above_threshold"] == (body["probability"] >= body["threshold"])
    assert "Not medical advice" in body["disclaimer"]


@pytest.mark.parametrize("field,value", [("Glucose", 500), ("Glucose", 0), ("Age", 18), ("BMI", 5),
                                         ("Insulin", 2000), ("DiabetesPedigreeFunction", 3.0), ("Pregnancies", -1)])
def test_out_of_range_is_rejected(client, field, value):
    r = client.post("/predict", json={**FULL, field: value})
    assert r.status_code == 422


def test_missing_required_and_unknown_fields_are_rejected(client):
    assert client.post("/predict", json={k: v for k, v in FULL.items() if k != "Glucose"}).status_code == 422
    assert client.post("/predict", json={**FULL, "Cholesterol": 200}).status_code == 422


def test_omitted_optional_fields_are_listed_as_imputed(client):
    r = client.post("/predict", json=REQUIRED_ONLY)
    assert r.status_code == 200
    assert sorted(r.json()["imputed"]) == ["DiabetesPedigreeFunction", "Insulin", "SkinThickness"]
    r = client.post("/predict", json={**REQUIRED_ONLY, "Insulin": 90})
    assert sorted(r.json()["imputed"]) == ["DiabetesPedigreeFunction", "SkinThickness"]


def test_high_risk_profile_scores_higher_than_low_risk(client):
    high = {"Pregnancies": 8, "Glucose": 190, "BloodPressure": 85, "BMI": 42, "DiabetesPedigreeFunction": 1.2, "Age": 55}
    low = {"Pregnancies": 0, "Glucose": 85, "BloodPressure": 66, "BMI": 22, "DiabetesPedigreeFunction": 0.2, "Age": 23}
    p_high = client.post("/predict", json=high).json()["probability"]
    p_low = client.post("/predict", json=low).json()["probability"]
    assert p_high > p_low
    assert p_high > 0.5 > p_low


def test_model_info_matches_saved_metrics(client):
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    info = client.get("/model-info").json()
    assert info["threshold"] == metrics["threshold"]["value"]
    assert info["metrics"]["test"]["roc_auc"] == metrics["test"]["roc_auc"]
    assert [f["name"] for f in info["features"]] == ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                                                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    glucose = next(f for f in info["features"] if f["name"] == "Glucose")
    assert (glucose["min"], glucose["max"], glucose["unit"]) == (40, 300, "mg/dL")


def test_cors_allows_localhost_but_not_any_origin(client):
    ok = client.get("/health", headers={"Origin": "http://localhost:7860"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:7860"
    other = client.get("/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in other.headers


def test_web_form_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Diabetes risk estimate" in r.text


def test_input_values_are_not_logged(client, caplog, capsys):
    marker = 187.123
    with caplog.at_level(logging.DEBUG):
        client.post("/predict", json={**FULL, "Glucose": marker})
    out = capsys.readouterr()
    assert str(marker) not in caplog.text + out.out + out.err
