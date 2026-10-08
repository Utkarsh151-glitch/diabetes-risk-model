---
title: Diabetes Risk Model
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
license: gpl-3.0
short_description: Leak-free diabetes risk model (Pima) with a web form
pinned: false
---

# Diabetes Risk Model

A logistic regression model trained on the Pima Indians Diabetes Database (768 women aged 21 or older of Pima Indian heritage), evaluated without data leakage and served by FastAPI with a web form.

- Web form: `/`
- API: `GET /health`, `GET /model-info`, `POST /predict` (docs at `/docs`)
- Code, data, evaluation and tests: https://github.com/Utkarsh151-glitch/diabetes-risk-model

**Not medical advice.** This is a statistical estimate from a small, single-population dataset. It cannot diagnose diabetes. Talk to a doctor.
