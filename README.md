# Early Prediction of Clinical Deterioration in Older Adults With Suspected Infection

Trained machine-learning models and inference code accompanying the manuscript
*"Early Prediction of Clinical Deterioration in Older Adults With Sepsis Admitted From the
Emergency Department to General Wards: A Multicenter Machine Learning Study."*

The models predict the risk of **early clinical deterioration within 48 hours** (a composite of
unplanned ICU transfer, mechanical ventilation, or death) among older adults (≥65 years) with
suspected infection admitted from the emergency department (ED) to a general ward.

> **Intended use.** This is a research artifact and clinical-decision-*support* tool. It is
> **not** a medical device and has not been prospectively validated for clinical use. It must
> not be used as the sole basis for any clinical decision. Predictions should always be
> interpreted alongside clinical judgment.

## Contents

```
models/
  catboost_missforest.cbm                 CatBoost model (primary)
  catboost_missforest_calibrator.joblib   isotonic calibrator for CatBoost
  xgb_missforest.json                     XGBoost model
  xgb_missforest_calibrator.joblib        isotonic calibrator for XGBoost
  missforest_pipeline_cat.joblib          fitted missingness-indicator + MissForest imputer + feature order
src/
  predict.py                              inference example (raw features -> calibrated risk -> tier)
  train_missforest.py                     training / preprocessing pipeline
data/
  operating_thresholds.csv                thresholds and performance at 90%/80% sensitivity and Youden index
requirements.txt
```

## Model performance (patient-level split)

| Model | Internal AUROC | External AUROC | External AUPRC |
|-------|:--------------:|:--------------:|:--------------:|
| CatBoost | 0.828 | 0.796 | 0.194 |
| XGBoost  | 0.827 | 0.798 | 0.198 |

External event rate ≈ 2%; AUPRC no-skill baseline ≈ 0.02. Probabilities are isotonic-calibrated
on a held-out validation set.

## Usage

```python
import pandas as pd
from src.predict import DeteriorationModel

# df_raw: one row per patient, columns = raw EHR variables used in training
# (arrival vitals *_a, pre-ward vitals *_in, labs, comorbidities, ED management).
df_raw = pd.read_csv("your_patients.csv")

model = DeteriorationModel(model="catboost")   # or "xgboost"
out = model.predict_tier(df_raw)               # -> calibrated_risk, risk_tier
print(out)
```

The full pipeline is: **raw features → missingness indicators + MissForest imputation →
gradient-boosting model → isotonic calibration → calibrated 48-hour risk → risk tier**.
Missing inputs are handled automatically by the fitted imputer; the model must always be applied
through `src/predict.py` (or the saved pipeline) so that preprocessing matches training exactly.

## Risk tiers

Tiers in `predict_tier` are derived from the external-cohort operating points in
`data/operating_thresholds.csv`:

- **High**: risk ≥ 80%-sensitivity threshold
- **Intermediate**: risk ≥ 90%-sensitivity threshold
- **Low**: below the 90%-sensitivity threshold

At ~2% prevalence, high sensitivity entails a substantial false-alert burden (see
`operating_thresholds.csv`, *alerts per true positive*); thresholds should be chosen locally.

## Installation

```bash
pip install -r requirements.txt
```

## Citation

Please cite the accompanying manuscript. A dataset/DOI archive (e.g., Zenodo) can be linked here
upon publication.

## License

Released for academic research use. See the manuscript and your institution's data-use terms.
