# Flood Warning System

AI-assisted flood risk warning and rainfall forecasting for Karnataka districts, with a Streamlit dashboard and geospatial risk visualization.

## Current Project Features

1. Data ingestion from Open-Meteo archive API for daily rainfall and soil moisture.
2. Multi-district preprocessing pipeline with engineered flood risk labels.
3. Random Forest risk classification model using rainfall and soil moisture features.
4. Time-series rainfall forecasting using Prophet with per-district annual aggregates.
5. Live dashboard with district selector, manual/live input modes, hotspot map layers, and historical context.
6. Prediction confidence display (class probabilities) when model supports `predict_proba`.
7. Automated tests and CI workflow for quick regression checks.

## Repository Structure

- `app.py`: Streamlit dashboard and live risk prediction workflow.
- `preprocess_data.py`: Data ingestion, cleaning, and risk-label generation.
- `train_model.py`: Model training, evaluation, and metrics artifact export.
- `forecast_rainfall.py`: District-level annual aggregation and Prophet forecasting.
- `requirements.txt`: Pinned dependencies.
- `tests/`: Unit tests for preprocessing and forecast data preparation.
- `.github/workflows/ci.yml`: CI pipeline that installs dependencies and runs tests.
- `data/`: Input datasets and generated preprocessed dataset.
- `models/`: Trained model and training metrics JSON.
- `output/`: Forecast CSV/plot outputs.

## Fix Log (Applied One by One)

### Fix 1: Forecasting schema mismatch (Done)

Problem:
Forecasting expected `ANNUAL` in preprocessed data, but preprocessing produced only daily rows.

Fix:
Updated `forecast_rainfall.py` to aggregate daily `Rainfall_mm` into annual totals per district and added validation for minimum historical years before running Prophet.

Result:
Forecasting no longer depends on a missing `ANNUAL` column.

### Fix 2: Soil moisture scale mismatch (Done)

Problem:
Model trained on soil moisture ratio values (0.0 to 1.0), while dashboard input was 0 to 100.

Fix:
Dashboard now converts UI percentage to ratio (`value / 100.0`) before prediction.

Result:
Inference input distribution matches training distribution.

### Fix 3: Time leakage in model validation (Done)

Problem:
Random split can overestimate performance for time-dependent weather data.

Fix:
Replaced random split with chronological 80/20 split in `train_model.py` and added balanced accuracy plus metrics export (`models/model_metrics.json`).

Result:
Evaluation now better reflects real future forecasting conditions.

### Fix 4: API reliability and error handling (Done)

Problem:
API calls had limited timeout/schema protection.

Fix:
Added request timeout and schema checks in dashboard live fetch, plus a retry-capable API fetch helper in preprocessing.

Result:
Better resilience to transient network/API issues.

### Fix 5: Single-district data limitation (Partially Improved)

Problem:
Training data pipeline originally pulled only Udupi.

Fix:
Preprocessing now supports multiple Karnataka districts via coordinate map, and dashboard includes district selector for live inference and historical context.

Result:
Project is now district-aware for selected configured districts.

Note:
Full statewide coverage still needs a complete district coordinate list and broader data collection strategy.

### Fix 6: Reproducibility and dependency management (Done)

Problem:
No pinned environment spec.

Fix:
Added `requirements.txt` with pinned package versions.

Result:
Faster and more reproducible setup.

### Fix 7: Tests and CI quality gate (Done)

Problem:
No automated tests/validation pipeline.

Fix:
Added unit tests for risk-label logic and annual aggregation prep, and added GitHub Actions CI workflow to run tests on push and PR.

Result:
Basic regression safety net is in place.

### Fix 8: Dashboard UX clarity (Done)

Problem:
No confidence visibility and limited traceability of live refresh.

Fix:
Added prediction confidence display and last live refresh timestamp, and improved map responsiveness using container width.

Result:
Better trust and readability for operators.

### Fix 9: Documentation mismatch (Done)

Problem:
Previous README described behaviors not aligned with actual data flow.

Fix:
Rewrote README to match current architecture and implementation.

Result:
Documentation now reflects real execution flow.

### Fix 10: Ground-truth flood label realism (Not Fully Solved)

Problem:
Current `Risk_Level` target is rule-engineered from weather thresholds.

Status:
This is a data availability problem, not just a code bug.

Recommended next step:
Integrate true historical flood incident labels from government/disaster records and retrain with observed outcomes.

## Project Analysis Summary

### Strengths

1. Clear end-to-end pipeline from ingestion to dashboard prediction.
2. Fast local iteration with lightweight model and scripts.
3. Good geospatial communication for hotspot awareness.

### Risks / Gaps Remaining

1. Ground-truth target labels are still synthetic.
2. District coverage remains limited to configured coordinates.
3. No scheduler yet for automated periodic refresh + retraining.
4. No integration/e2e dashboard tests yet.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Sequence

1. Build/refresh dataset:

```bash
python3 preprocess_data.py
```

1. Train model:

```bash
python3 train_model.py
```

1. Generate rainfall forecast:

```bash
python3 forecast_rainfall.py
```

1. Start dashboard:

```bash
streamlit run app.py
```

1. Run tests:

```bash
pytest -q tests
```

## Validation

- Local tests: `3 passed`
- Model metrics artifact: `models/model_metrics.json`

## Suggested Next Enhancements

1. Add a scheduled job for periodic data refresh + retraining.
2. Add alert channels (SMS/Email/WhatsApp) for `High` risk predictions.
3. Add feature engineering from lag rainfall, rolling windows, and upstream catchment signals.
4. Add calibration plots and threshold tuning for operational deployment.
