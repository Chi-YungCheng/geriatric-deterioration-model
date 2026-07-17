"""
Inference example for the geriatric early-deterioration model.

Pipeline: raw EHR features -> missingness indicators + MissForest imputation
          -> gradient-boosting model -> isotonic calibration -> calibrated risk
          -> risk tier (from operating_thresholds.csv).

Usage:
    from predict import DeteriorationModel
    model = DeteriorationModel(model="catboost")     # or "xgboost"
    risk = model.predict_risk(df_raw)                # df_raw: pandas DataFrame of raw features
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


class DeteriorationModel:
    def __init__(self, model: str = "catboost"):
        self.model_name = model.lower()
        pipe = joblib.load(MODELS / "missforest_pipeline_cat.joblib")
        self.indicator = pipe["indicator"]
        # tolerate missing values at inference in columns that were complete during fitting
        self.indicator.error_on_new = False
        self.imputer = pipe["imputer"]
        self.impute_cols = pipe["impute_cols"]     # base feature columns (incl. PULSE_delta_v1)
        self.icols = pipe["icols"]                 # missing-indicator column names
        self.feature_order = pipe["feature_order"] # final column order fed to the model

        if self.model_name == "catboost":
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier()
            self.model.load_model(str(MODELS / "catboost_missforest.cbm"))
            self.calibrator = joblib.load(MODELS / "catboost_missforest_calibrator.joblib")
        elif self.model_name == "xgboost":
            from xgboost import XGBClassifier
            self.model = XGBClassifier()
            self.model.load_model(str(MODELS / "xgb_missforest.json"))
            self.calibrator = joblib.load(MODELS / "xgb_missforest_calibrator.joblib")
        else:
            raise ValueError("model must be 'catboost' or 'xgboost'")

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        X = df.copy()
        # engineered feature used in training
        if "PULSE_delta_v1" not in X.columns:
            X["PULSE_delta_v1"] = pd.to_numeric(X.get("PULSE_in"), errors="coerce") - pd.to_numeric(
                X.get("PULSE_a"), errors="coerce"
            )
        # ensure all required base columns exist
        for c in self.impute_cols:
            if c not in X.columns:
                X[c] = np.nan
            X[c] = pd.to_numeric(X[c], errors="coerce")
        base = X[self.impute_cols]
        ind = pd.DataFrame(self.indicator.transform(base), columns=self.icols, index=X.index)
        aug = pd.concat([base.reset_index(drop=True), ind.reset_index(drop=True)], axis=1)
        aug[self.impute_cols] = self.imputer.transform(aug[self.impute_cols])
        return aug[self.feature_order].to_numpy()

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Return calibrated 48-hour deterioration risk (0-1) for each row."""
        Xn = self._prepare(df)
        raw = self.model.predict_proba(Xn)[:, 1]
        return self.calibrator.transform(raw)

    def predict_tier(self, df: pd.DataFrame, thresholds_csv: str = None) -> pd.DataFrame:
        """Return calibrated risk plus a risk tier based on operating_thresholds.csv."""
        risk = self.predict_risk(df)
        thr_path = Path(thresholds_csv) if thresholds_csv else ROOT / "data" / "operating_thresholds.csv"
        thr = pd.read_csv(thr_path)
        row = thr[(thr.model.str.lower() == self.model_name) & (thr.set == "External")]
        t90 = float(row[row.operating_point == "90% sensitivity"].threshold.iloc[0])
        t80 = float(row[row.operating_point == "80% sensitivity"].threshold.iloc[0])
        tier = np.where(risk >= t80, "High", np.where(risk >= t90, "Intermediate", "Low"))
        return pd.DataFrame({"calibrated_risk": risk, "risk_tier": tier})


if __name__ == "__main__":
    # illustrative demo (sparse input; missing variables are imputed automatically)
    demo = pd.DataFrame([
        {"AGE": 82, "SEX": 1, "PULSE_a": 120, "SBP_a": 90, "RR_a": 28,
         "GCSE_a": 3, "GCSV_a": 4, "GCSM_a": 5, "Creatinine": 2.5,
         "PULSE_in": 125, "SBP_in": 85, "RR_in": 30},
        {"AGE": 70, "SEX": 0, "PULSE_a": 80, "SBP_a": 130, "RR_a": 16,
         "GCSE_a": 4, "GCSV_a": 5, "GCSM_a": 6, "Creatinine": 0.9,
         "PULSE_in": 78, "SBP_in": 132, "RR_in": 15},
    ])
    for name in ("catboost", "xgboost"):
        print(f"== {name} ==")
        print(DeteriorationModel(name).predict_tier(demo).to_string(index=False))
