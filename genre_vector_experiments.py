"""
Expériences de migration "vecteurs de genres".

Ce script compare trois scénarios sur un protocole reproductible:
1) baseline_ui_zero_genres : simulation de l'UI historique (features genre à 0)
2) phase1_multihot_binary   : genres en multi-hot binaire
3) phase2_multihot_weighted : genres en vecteur pondéré

Il produit:
- un rapport Markdown (`genre_vector_experiment_report.md`)
- un export JSON (`genre_vector_experiment_results.json`)
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

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


def _load_base_dataframe() -> pd.DataFrame:
    data_path = kagglehub.dataset_download("jacopoferretti/idmb-movies-user-friendly")
    df = pd.read_csv(f"{data_path}/MOVIES.csv")

    df["genre_list"] = df["genre"].apply(_parse_list)
    df["genre_target"] = df["genre_list"].apply(lambda genres: genres[0] if genres else None)
    df = df[df["genre_target"].isin(ALLOWED_GENRES)].copy()
    df = df[df["vote_count"].fillna(0) >= 50].copy()

    # Reprise de l'hypothèse documentée: imputation runtime par médiane intra-genre.
    group_runtime_medians = df.groupby("genre_target")["runtime"].transform("median")
    df["runtime"] = df["runtime"].fillna(group_runtime_medians)
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())

    df["companies_list"] = df["companies"].apply(_parse_list)
    df["countries_list"] = df["countries"].apply(_parse_list)
    df["has_collection"] = (~df["belongs_to_collection"].isin(["not available", "", np.nan])).astype(int)

    return df


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


@dataclass
class ScenarioResult:
    name: str
    model_name: str
    metrics: dict[str, float]


def _genre_weights_from_position(genre_list: Iterable[str]) -> dict[str, float]:
    # Pondération recommandée par la roadmap: décroissance par position.
    decay = [1.0, 0.6, 0.3]
    weighted = {}
    for idx, genre_name in enumerate(genre_list):
        weighted[genre_name] = decay[idx] if idx < len(decay) else 0.2
    return weighted


def _build_feature_frame(
    df: pd.DataFrame,
    artifacts: dict,
    strategy: str,
) -> pd.DataFrame:
    all_features = artifacts["all_features"]
    top_genres = artifacts["top_genres"]
    top_companies = artifacts["top_companies"]
    top_countries = artifacts["top_countries"]
    le_lang = artifacts["le_lang"]
    le_month = artifacts["le_month"]
    le_season = artifacts["le_season"]
    le_day = artifacts["le_day"]
    le_homepage = artifacts["le_homepage"]

    records = []
    for row in df.itertuples(index=False):
        feature_values: dict[str, float | int] = {}

        feature_values["vote_count"] = float(getattr(row, "vote_count", 0.0))
        feature_values["vote_average"] = float(getattr(row, "vote_average", 0.0))
        feature_values["popularity"] = float(getattr(row, "popularity", 0.0))
        feature_values["runtime"] = float(getattr(row, "runtime", 0.0))
        feature_values["year"] = float(getattr(row, "year", 0.0))

        feature_values["lang_enc"] = _safe_label_encode(le_lang, getattr(row, "original_language", None))
        feature_values["month_enc"] = _safe_label_encode(le_month, str(getattr(row, "month", "")))
        feature_values["season_enc"] = _safe_label_encode(le_season, str(getattr(row, "season", "")))
        feature_values["day_enc"] = _safe_label_encode(le_day, str(getattr(row, "day_of_week", "")))
        feature_values["homepage_enc"] = _safe_label_encode(le_homepage, getattr(row, "has_homepage", None))
        feature_values["has_collection"] = int(getattr(row, "has_collection", 0))

        genre_values = {}
        if strategy == "baseline_ui_zero_genres":
            genre_values = {genre_name: 0.0 for genre_name in top_genres}
        elif strategy == "phase1_multihot_binary":
            # On ne conserve que les genres secondaires pour éviter toute fuite de cible.
            row_genres = set(getattr(row, "genre_list", [])[1:])
            genre_values = {genre_name: (1.0 if genre_name in row_genres else 0.0) for genre_name in top_genres}
        elif strategy == "phase2_multihot_weighted":
            # Pondération appliquée uniquement aux genres secondaires (position relative).
            weights = _genre_weights_from_position(getattr(row, "genre_list", [])[1:])
            genre_values = {genre_name: float(weights.get(genre_name, 0.0)) for genre_name in top_genres}
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")

        for genre_name in top_genres:
            feature_values[f"has_{genre_name}"] = genre_values[genre_name]

        row_companies = set(getattr(row, "companies_list", []))
        for company_name in top_companies:
            feature_values[f"company_{company_name}"] = 1.0 if company_name in row_companies else 0.0

        row_countries = set(getattr(row, "countries_list", []))
        for country_name in top_countries:
            feature_values[f"country_{country_name}"] = 1.0 if country_name in row_countries else 0.0

        for feature_name in all_features:
            feature_values.setdefault(feature_name, 0.0)

        records.append({feature_name: feature_values[feature_name] for feature_name in all_features})

    return pd.DataFrame(records, columns=all_features)


def _run_scenario(
    scenario_name: str,
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> ScenarioResult:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return ScenarioResult(
        name=scenario_name,
        model_name=model_name,
        metrics=_metric_bundle(y_test, y_pred),
    )


def _generate_report(
    output_path: Path,
    results_path: Path,
    baseline_metrics: dict[str, float],
    results: list[ScenarioResult],
    gate_min_gain: float,
) -> None:
    best_by_f1_macro = max(results, key=lambda item: item.metrics["f1_macro"])
    weighted_nb = next(item for item in results if item.name == "phase2_multihot_weighted" and item.model_name == "GaussianNB")
    gain_vs_phase1 = weighted_nb.metrics["f1_macro"] - next(
        item.metrics["f1_macro"]
        for item in results
        if item.name == "phase1_multihot_binary" and item.model_name == "GaussianNB"
    )

    if gain_vs_phase1 >= gate_min_gain:
        gate_decision = (
            "Le gain du vecteur pondéré est suffisant: priorité à la stabilisation Phase 2. "
            "La phase embeddings reste désactivée pour l'instant."
        )
    else:
        gate_decision = (
            "Le gain du vecteur pondéré est insuffisant: ouvrir un spike embeddings (Phase 3) "
            "dans une branche dédiée et comparer à la régression logistique."
        )

    lines = [
        "# Rapport d'expériences — Vecteurs de genres",
        "",
        "## Baseline verrouillée (artefacts actuels)",
        f"- Accuracy: {baseline_metrics['accuracy']:.4f}",
        f"- F1 weighted: {baseline_metrics['f1_weighted']:.4f}",
        f"- F1 macro: {baseline_metrics['f1_macro']:.4f}",
        "",
        "## Résultats de migration",
        "",
        "| Scénario | Modèle | Accuracy | F1 weighted | F1 macro |",
        "|---|---|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| `{result.name}` | `{result.model_name}` | "
            f"{result.metrics['accuracy']:.4f} | {result.metrics['f1_weighted']:.4f} | {result.metrics['f1_macro']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Décision de gate Phase 3 (embeddings)",
            f"- Seuil de décision (gain F1 macro pondéré vs phase 1): {gate_min_gain:.4f}",
            f"- Gain observé: {gain_vs_phase1:.4f}",
            f"- Décision: {gate_decision}",
            "",
            "## Meilleur scénario observé",
            f"- `{best_by_f1_macro.name}` avec `{best_by_f1_macro.model_name}` (F1 macro = {best_by_f1_macro.metrics['f1_macro']:.4f})",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    raw = {
        "baseline_metrics": baseline_metrics,
        "results": [
            {
                "scenario": result.name,
                "model": result.model_name,
                "metrics": result.metrics,
            }
            for result in results
        ],
        "gate_min_gain": gate_min_gain,
        "phase3_decision": gate_decision,
    }
    results_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    artifacts = joblib.load("model_artifacts.pkl")
    baseline_metrics = artifacts["all_model_metrics"][artifacts["best_model_name"]]

    df = _load_base_dataframe()
    y = artifacts["label_encoder"].transform(df["genre_target"])

    scenario_frames = {
        "baseline_ui_zero_genres": _build_feature_frame(df, artifacts, "baseline_ui_zero_genres"),
        "phase1_multihot_binary": _build_feature_frame(df, artifacts, "phase1_multihot_binary"),
        "phase2_multihot_weighted": _build_feature_frame(df, artifacts, "phase2_multihot_weighted"),
    }

    raw_test_size = artifacts.get("test_size", 0.2)
    if isinstance(raw_test_size, (int, float)) and raw_test_size >= 1:
        total = float(artifacts.get("train_size", 0)) + float(artifacts.get("test_size", 0))
        test_size = (float(raw_test_size) / total) if total > 0 else 0.2
    else:
        test_size = float(raw_test_size)

    gate_min_gain = 0.005
    all_results: list[ScenarioResult] = []

    for scenario_name, X_df in scenario_frames.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values.astype(np.float64),
            y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        all_results.append(
            _run_scenario(
                scenario_name=scenario_name,
                model_name="GaussianNB",
                model=GaussianNB(),
                X_train=X_train_scaled,
                X_test=X_test_scaled,
                y_train=y_train,
                y_test=y_test,
            )
        )

        if scenario_name == "phase2_multihot_weighted":
            all_results.append(
                _run_scenario(
                    scenario_name=scenario_name,
                    model_name="LogisticRegression(multinomial)",
                    model=LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                    X_train=X_train_scaled,
                    X_test=X_test_scaled,
                    y_train=y_train,
                    y_test=y_test,
                )
            )

    report_path = Path("genre_vector_experiment_report.md")
    results_path = Path("genre_vector_experiment_results.json")
    _generate_report(
        output_path=report_path,
        results_path=results_path,
        baseline_metrics=baseline_metrics,
        results=all_results,
        gate_min_gain=gate_min_gain,
    )

    print(f"Rapport généré: {report_path}")
    print(f"Résultats JSON: {results_path}")


if __name__ == "__main__":
    main()
