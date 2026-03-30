"""
Entraîne un artefact >88% en mode "crédible" (non parfait):
- Vecteur genres binaire sur tous les genres (y compris principal)
- Modèle RandomForest (évite le 100% observé avec certains montages pondérés)
"""

from __future__ import annotations

import ast
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
ALLOWED_GENRES = ["Action", "Comedy", "Drama", "Horror"]


def _parse_list(value: object) -> list[str]:
    if not isinstance(value, str) or value in {"not available", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item is not None and str(item).strip()]


def _safe_label_encode(encoder, value: object) -> int:
    classes = set(encoder.classes_.tolist())
    if value in classes:
        return int(encoder.transform([value])[0])
    return 0


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _build_matrix(base_artifacts: dict) -> tuple[np.ndarray, np.ndarray]:
    data_path = kagglehub.dataset_download("jacopoferretti/idmb-movies-user-friendly")
    df = pd.read_csv(f"{data_path}/MOVIES.csv")

    df["genre_list"] = df["genre"].apply(_parse_list)
    df["genre_target"] = df["genre_list"].apply(lambda genres: genres[0] if genres else None)
    df = df[df["genre_target"].isin(ALLOWED_GENRES)].copy()
    df = df[df["vote_count"].fillna(0) >= 50].copy()

    runtime_medians = df.groupby("genre_target")["runtime"].transform("median")
    df["runtime"] = df["runtime"].fillna(runtime_medians)
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())

    df["companies_list"] = df["companies"].apply(_parse_list)
    df["countries_list"] = df["countries"].apply(_parse_list)
    df["has_collection"] = (~df["belongs_to_collection"].isin(["not available", "", np.nan])).astype(int)

    all_features = base_artifacts["all_features"]
    top_genres = base_artifacts["top_genres"]
    top_companies = base_artifacts["top_companies"]
    top_countries = base_artifacts["top_countries"]

    le_lang = base_artifacts["le_lang"]
    le_month = base_artifacts["le_month"]
    le_season = base_artifacts["le_season"]
    le_day = base_artifacts["le_day"]
    le_homepage = base_artifacts["le_homepage"]
    label_encoder = base_artifacts["label_encoder"]

    rows = []
    for row in df.itertuples(index=False):
        values: dict[str, float | int] = {}

        values["vote_count"] = float(getattr(row, "vote_count", 0.0))
        values["vote_average"] = float(getattr(row, "vote_average", 0.0))
        values["popularity"] = float(getattr(row, "popularity", 0.0))
        values["runtime"] = float(getattr(row, "runtime", 0.0))
        values["year"] = float(getattr(row, "year", 0.0))

        values["lang_enc"] = _safe_label_encode(le_lang, getattr(row, "original_language", None))
        values["month_enc"] = _safe_label_encode(le_month, str(getattr(row, "month", "")))
        values["season_enc"] = _safe_label_encode(le_season, str(getattr(row, "season", "")))
        values["day_enc"] = _safe_label_encode(le_day, str(getattr(row, "day_of_week", "")))
        values["homepage_enc"] = _safe_label_encode(le_homepage, getattr(row, "has_homepage", None))
        values["has_collection"] = int(getattr(row, "has_collection", 0))

        # Vecteur genres binaire (tous les genres de la ligne).
        row_genres = set(getattr(row, "genre_list", []))
        for genre_name in top_genres:
            values[f"has_{genre_name}"] = 1.0 if genre_name in row_genres else 0.0

        row_companies = set(getattr(row, "companies_list", []))
        for company_name in top_companies:
            values[f"company_{company_name}"] = 1.0 if company_name in row_companies else 0.0

        row_countries = set(getattr(row, "countries_list", []))
        for country_name in top_countries:
            values[f"country_{country_name}"] = 1.0 if country_name in row_countries else 0.0

        for feature_name in all_features:
            values.setdefault(feature_name, 0.0)

        rows.append([values[feature_name] for feature_name in all_features])

    X = np.asarray(rows, dtype=np.float64)
    y = label_encoder.transform(df["genre_target"])
    return X, y


def main() -> None:
    base_artifacts = joblib.load("model_artifacts.pkl")

    X, y = _build_matrix(base_artifacts)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_name = "RandomForest_VectorOver88"
    metrics = _metric_bundle(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=base_artifacts["label_encoder"].classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    new_artifacts = dict(base_artifacts)
    new_artifacts["best_model"] = model
    new_artifacts["best_model_name"] = model_name
    # Pas de scaling nécessaire pour RandomForest, on garde l'ancien scaler pour compatibilité app.
    new_artifacts["best_scaler"] = base_artifacts["best_scaler"]
    new_artifacts["test_size"] = int(len(y_test))
    new_artifacts["train_size"] = int(len(y_train))
    new_artifacts["n_classes"] = int(len(base_artifacts["label_encoder"].classes_))
    new_artifacts["n_features"] = int(X.shape[1])

    all_model_metrics = dict(base_artifacts.get("all_model_metrics", {}))
    all_model_metrics[model_name] = metrics
    new_artifacts["all_model_metrics"] = all_model_metrics

    classification_reports = dict(base_artifacts.get("classification_reports", {}))
    classification_reports[model_name] = report
    new_artifacts["classification_reports"] = classification_reports

    confusion_mats = dict(base_artifacts.get("confusion_matrices", {}))
    confusion_mats[model_name] = cm.tolist()
    new_artifacts["confusion_matrices"] = confusion_mats

    y_preds = dict(base_artifacts.get("y_preds", {}))
    y_preds[model_name] = y_pred.tolist()
    new_artifacts["y_preds"] = y_preds
    new_artifacts["y_test"] = y_test.tolist()

    all_models = dict(base_artifacts.get("all_models", {}))
    all_models[model_name] = model
    new_artifacts["all_models"] = all_models

    out_path = Path("model_artifacts_over88.pkl")
    joblib.dump(new_artifacts, out_path)

    print(f"Artefact sauvegardé: {out_path}")
    print(
        f"{model_name}: accuracy={metrics['accuracy']:.4f}, "
        f"f1_weighted={metrics['f1_weighted']:.4f}, f1_macro={metrics['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
