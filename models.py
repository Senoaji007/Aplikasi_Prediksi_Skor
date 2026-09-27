"""
models.py
=========
Eksperimen pelatihan & evaluasi model (Bab 5.1 poin 6-7 dan Bab 6.1
Metodologi Penelitian):
  - Model utama   : XGBoost (klasifikasi hasil H/D/A + regresi skor)
  - Benchmark     : Random Forest
  - Baseline      : Logistic Regression (klasifikasi)
  - Pembagian data: time-based split (kronologis) untuk mencegah
    kebocoran data dari masa depan ke masa lalu.
  - Interpretabilitas: SHAP (SHapley Additive exPlanations).
"""

from __future__ import annotations

import hashlib
import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from features import FEATURE_COLUMNS

RANDOM_STATE = 42
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".cache_model_artifacts.joblib")


def _classification_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _dataset_fingerprint(feature_df: pd.DataFrame, test_frac: float) -> str:
    """
    Sidik jari ringan dari data fitur & konfigurasi pelatihan, dipakai untuk
    memvalidasi cache model di disk -- kalau datanya persis sama dengan
    training terakhir, model tidak perlu dilatih ulang dari nol.
    """
    key_parts = [
        str(len(feature_df)),
        str(feature_df["Date"].max()) if "Date" in feature_df.columns and len(feature_df) else "",
        str(test_frac),
        ",".join(FEATURE_COLUMNS),
    ]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()


def _load_model_cache(fingerprint: str) -> dict | None:
    if not os.path.exists(MODEL_CACHE_PATH):
        return None
    try:
        cached = joblib.load(MODEL_CACHE_PATH)
        if cached.get("fingerprint") != fingerprint:
            return None
        artifacts = cached["artifacts"]
        # Explainer SHAP dibangun ulang (cepat) alih-alih di-pickle, untuk
        # menghindari ketidakcocokan versi shap antar proses/deploy.
        artifacts["explainer"] = shap.TreeExplainer(artifacts["xgb_clf"])
        return artifacts
    except Exception:
        return None


def _save_model_cache(fingerprint: str, artifacts: dict) -> None:
    try:
        to_save = {k: v for k, v in artifacts.items() if k != "explainer"}
        joblib.dump({"fingerprint": fingerprint, "artifacts": to_save}, MODEL_CACHE_PATH)
    except Exception:
        pass  # Cache disk bersifat opsional; kegagalan menulis tidak fatal


@st.cache_resource(show_spinner=False)
def train_and_evaluate(feature_df: pd.DataFrame, test_frac: float = 0.15):
    """
    Melatih model utama (XGBoost) dan model benchmark/baseline (Random
    Forest, Logistic Regression) menggunakan time-based split, lalu
    mengevaluasinya dengan metrik Accuracy/Precision/Recall/F1 (klasifikasi
    hasil pertandingan) dan RMSE/MAE (regresi skor).

    Hasil pelatihan disimpan ke cache disk (lihat MODEL_CACHE_PATH); bila
    data & konfigurasi tidak berubah, proses berikutnya (mis. saat aplikasi
    bangun dari mode tidur) cukup memuat model dari disk tanpa melatih ulang.
    """
    fingerprint = _dataset_fingerprint(feature_df, test_frac)
    cached_artifacts = _load_model_cache(fingerprint)
    if cached_artifacts is not None:
        return cached_artifacts

    df = feature_df.dropna(subset=["FTHG", "FTAG", "FTR"]).sort_values("Date").reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    label_encoder = LabelEncoder()
    y_cls = label_encoder.fit_transform(df["FTR"])
    y_home, y_away = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()

    n = len(df)
    split = max(int(n * (1 - test_frac)), 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_cls_train, y_cls_test = y_cls[:split], y_cls[split:]
    yh_train, yh_test = y_home[:split], y_home[split:]
    ya_train, ya_test = y_away[:split], y_away[split:]

    # ---- Klasifikasi hasil pertandingan (H/D/A) ----
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    xgb_clf.fit(X_train, y_cls_train)

    rf_clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, random_state=RANDOM_STATE,
    )
    rf_clf.fit(X_train, y_cls_train)

    logreg = LogisticRegression(max_iter=1000, C=1.0)
    logreg.fit(X_train, y_cls_train)

    metrics = {
        "XGBoost": _classification_metrics(y_cls_test, xgb_clf.predict(X_test)),
        "Random Forest": _classification_metrics(y_cls_test, rf_clf.predict(X_test)),
        "Logistic Regression": _classification_metrics(y_cls_test, logreg.predict(X_test)),
    }

    # ---- Regresi skor (FTHG & FTAG) ----
    xgb_home = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    xgb_home.fit(X_train, yh_train)
    xgb_away = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    xgb_away.fit(X_train, ya_train)

    rf_home = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    rf_home.fit(X_train, yh_train)
    rf_away = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=RANDOM_STATE)
    rf_away.fit(X_train, ya_train)

    def _score_metrics(model_h, model_a):
        ph, pa = model_h.predict(X_test), model_a.predict(X_test)
        y_all = np.concatenate([yh_test, ya_test])
        p_all = np.concatenate([ph, pa])
        rmse = float(np.sqrt(mean_squared_error(y_all, p_all)))
        mae = float(mean_absolute_error(y_all, p_all))
        return {"RMSE": rmse, "MAE": mae}

    score_metrics = {
        "XGBoost": _score_metrics(xgb_home, xgb_away),
        "Random Forest": _score_metrics(rf_home, rf_away),
    }

    explainer = shap.TreeExplainer(xgb_clf)

    artifacts = {
        "xgb_clf": xgb_clf, "rf_clf": rf_clf, "logreg": logreg,
        "xgb_home": xgb_home, "xgb_away": xgb_away,
        "label_encoder": label_encoder, "medians": medians,
        "metrics": metrics, "score_metrics": score_metrics,
        "explainer": explainer,
        "n_train": len(X_train), "n_test": len(X_test),
    }
    _save_model_cache(fingerprint, artifacts)
    return artifacts


def predict_match(artifacts: dict, snapshot: dict) -> dict:
    """
    Menjalankan model utama (XGBoost) untuk satu pertandingan baru:
    memprediksi skor (regresi) dan probabilitas hasil H/D/A (klasifikasi),
    lengkap dengan nilai SHAP untuk kelas hasil yang paling mungkin.
    """
    x = pd.DataFrame([snapshot])[FEATURE_COLUMNS].fillna(artifacts["medians"])

    proba = artifacts["xgb_clf"].predict_proba(x)[0]
    classes = artifacts["label_encoder"].inverse_transform(np.arange(len(proba)))
    proba_map = dict(zip(classes, proba))

    home_goals = max(0, int(round(float(artifacts["xgb_home"].predict(x)[0]))))
    away_goals = max(0, int(round(float(artifacts["xgb_away"].predict(x)[0]))))

    predicted_class_idx = int(np.argmax(proba))
    shap_values = _extract_shap_for_class(artifacts["explainer"], x, predicted_class_idx)

    return {
        "predicted_score": (home_goals, away_goals),
        "win_probability": proba_map,
        "shap_values": shap_values,
        "feature_row": x.iloc[0],
    }


def _extract_shap_for_class(explainer, x: pd.DataFrame, class_idx: int) -> pd.Series:
    """Mengekstrak nilai SHAP untuk satu kelas, menangani beberapa versi API shap."""
    raw = explainer.shap_values(x)
    if isinstance(raw, list):
        sv = np.array(raw[class_idx])[0]
    else:
        arr = np.array(raw)
        if arr.ndim == 3:  # (n_samples, n_features, n_classes)
            sv = arr[0, :, class_idx]
        else:
            sv = arr[0]
    return pd.Series(sv, index=x.columns)
