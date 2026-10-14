"""Classifier factories (fixed hyper-parameters, single-threaded for outer joblib parallelism)."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_xgb(params: dict, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["learning_rate"], tree_method="hist", n_jobs=1,
        random_state=seed, eval_metric="logloss", verbosity=0,
    )


def make_model(name: str, cfg: dict, seed: int):
    p = cfg["classifiers"].get(name, {})
    if name == "xgboost":
        return make_xgb(p, seed)
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=p["n_estimators"], n_jobs=1, random_state=seed)
    if name == "mlp":
        # scaler is part of the pipeline -> fitted on source training data only
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=tuple(p["hidden_layer_sizes"]), activation="relu",
                          early_stopping=True, max_iter=p["max_iter"], random_state=seed),
        )
    raise ValueError(f"unknown classifier {name}")
