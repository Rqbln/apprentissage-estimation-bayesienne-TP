"""
Streamlit App — Prédiction du Genre des Films
============================================
Interface graphique interactive pour explorer le dataset, prédire le genre
d'un film et analyser les performances du modèle.

Le modèle affiché est chargé dynamiquement depuis un artefact .pkl.

Lancer avec : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import ast
import json
import os
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Genre Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des artefacts ──────────────────────────────────────────────────
DEFAULT_ARTIFACT_PATH = "model_artifacts_notebook_canonical.pkl"
ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH)


def _safe_label_encode(encoder: LabelEncoder, value):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0


def _season_from_month(month_value: int) -> str:
    if month_value in (12, 1, 2):
        return "Winter"
    if month_value in (3, 4, 5):
        return "Spring"
    if month_value in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@st.cache_resource
def load_artifacts(artifact_path: str, artifacts_mtime: float):
    """Charge les artefacts depuis un fichier canonique."""
    return joblib.load(artifact_path)


@st.cache_data
def load_dataset(target_genres: tuple[str, ...] = tuple(), min_votes: int = 50):
    """Charge et prépare une version transformée du dataset, alignée avec le notebook."""
    import kagglehub
    path = kagglehub.dataset_download("jacopoferretti/idmb-movies-user-friendly")
    df = pd.read_csv(f"{path}/MOVIES.csv")

    def parse_list(value):
        if not isinstance(value, str) or value in {"not available", "[]", "", "nan"}:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if item is not None and str(item).strip()]
        except Exception:
            return []
        return []

    # Colonnes transformées dans l'esprit du notebook
    df["genre_list"] = df["genre"].apply(parse_list)
    df["companies_list"] = df["companies"].apply(parse_list)
    df["countries_list"] = df["countries"].apply(parse_list)
    df["first_genre"] = df["genre_list"].apply(lambda x: x[0] if len(x) > 0 else None)
    df["genre_target"] = df["first_genre"]
    df["n_genres"] = df["genre_list"].apply(len)
    df["has_collection"] = (~df["belongs_to_collection"].isin(["not available", "", np.nan])).astype(int)

    # Conserve toutes les classes si target_genres est vide.
    df = df[df["genre_target"].notna()].copy()
    if len(target_genres) > 0:
        df = df[df["genre_target"].isin(target_genres)].copy()

    # Nettoyages proches du pipeline notebook
    if "runtime" in df.columns:
        runtime_by_genre = df.groupby("genre_target")["runtime"].transform("median")
        df["runtime"] = df["runtime"].fillna(runtime_by_genre)
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())

    if "vote_count" in df.columns:
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
        # Aligne l'exploration avec le pipeline d'entrainement (train_over88_artifact.py)
        df = df[df["vote_count"] >= float(min_votes)].copy()
    if "vote_average" in df.columns:
        df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)

    return df


@st.cache_resource
def build_runtime_artifacts_19(min_votes: int = 50, n_target_classes: int = 19):
    """Construit un pipeline canonique aligné sur les phases finales du notebook."""
    df = load_dataset(tuple(), min_votes=min_votes).copy()

    release_date = pd.to_datetime(df.get("release_date"), errors="coerce")
    df["month"] = release_date.dt.month.fillna(1).astype(int).astype(str)
    df["season"] = release_date.dt.month.fillna(1).astype(int).apply(_season_from_month)
    df["day_of_week"] = release_date.dt.dayofweek.fillna(0).astype(int).astype(str)
    if "has_homepage" not in df.columns:
        homepage_raw = df.get("homepage", "")
        df["has_homepage"] = (~pd.Series(homepage_raw).fillna("").astype(str).isin(["", "not available", "nan"]))
    df["has_homepage"] = df["has_homepage"].astype(str)

    top_target_classes = df["genre_target"].value_counts().head(n_target_classes).index.tolist()
    df = df[df["genre_target"].isin(top_target_classes)].copy()

    top_genres = df["genre_list"].explode().dropna().astype(str).value_counts().head(20).index.tolist()
    top_companies = df["companies_list"].explode().dropna().astype(str).value_counts().head(30).index.tolist()
    top_countries = df["countries_list"].explode().dropna().astype(str).value_counts().head(20).index.tolist()
    top_languages = df["original_language"].fillna("unknown").astype(str).value_counts().head(15).index.tolist()

    numeric_features = ["vote_count", "vote_average", "popularity", "runtime", "year"]
    encoded_features = ["lang_enc", "month_enc", "season_enc", "day_enc", "homepage_enc", "has_collection"]
    genre_features = [f"has_{g}" for g in top_genres]
    company_features = [f"company_{c}" for c in top_companies]
    country_features = [f"country_{c}" for c in top_countries]
    all_features = numeric_features + encoded_features + genre_features + company_features + country_features

    le_lang = LabelEncoder().fit(df["original_language"].fillna("unknown").astype(str))
    le_month = LabelEncoder().fit(df["month"].astype(str))
    le_season = LabelEncoder().fit(df["season"].astype(str))
    le_day = LabelEncoder().fit(df["day_of_week"].astype(str))
    le_homepage = LabelEncoder().fit(df["has_homepage"].astype(str))
    label_encoder = LabelEncoder().fit(df["genre_target"].astype(str))

    rows = []
    for row in df.itertuples(index=False):
        fv = {}
        fv["vote_count"] = float(getattr(row, "vote_count", 0.0))
        fv["vote_average"] = float(getattr(row, "vote_average", 0.0))
        fv["popularity"] = float(getattr(row, "popularity", 0.0))
        fv["runtime"] = float(getattr(row, "runtime", 0.0))
        fv["year"] = float(getattr(row, "year", 0.0))

        fv["lang_enc"] = _safe_label_encode(le_lang, str(getattr(row, "original_language", "unknown")))
        fv["month_enc"] = _safe_label_encode(le_month, str(getattr(row, "month", "1")))
        fv["season_enc"] = _safe_label_encode(le_season, str(getattr(row, "season", "Winter")))
        fv["day_enc"] = _safe_label_encode(le_day, str(getattr(row, "day_of_week", "0")))
        fv["homepage_enc"] = _safe_label_encode(le_homepage, str(getattr(row, "has_homepage", "False")))
        fv["has_collection"] = int(getattr(row, "has_collection", 0))

        genres_secondary = set((getattr(row, "genre_list", []) or [])[1:])
        for g in top_genres:
            fv[f"has_{g}"] = 1.0 if g in genres_secondary else 0.0

        companies = set(getattr(row, "companies_list", []) or [])
        for c in top_companies:
            fv[f"company_{c}"] = 1.0 if c in companies else 0.0

        countries = set(getattr(row, "countries_list", []) or [])
        for c in top_countries:
            fv[f"country_{c}"] = 1.0 if c in countries else 0.0

        rows.append(fv)

    X_df = pd.DataFrame(rows, columns=all_features).fillna(0.0)
    y = label_encoder.transform(df["genre_target"].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values.astype(np.float64),
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler_standard = StandardScaler().fit(X_train)
    scaler_minmax = MinMaxScaler().fit(X_train)

    X_train_std = scaler_standard.transform(X_train)
    X_test_std = scaler_standard.transform(X_test)
    X_train_mm = scaler_minmax.transform(X_train)
    X_test_mm = scaler_minmax.transform(X_test)

    ensemble_name = "NotebookEnsemble_Phase4B"

    # Split interne type Phase 4B pour tuning/weights/thresholds.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    sc_std = StandardScaler().fit(X_tr)
    sc_mm = MinMaxScaler().fit(X_tr)
    X_tr_std, X_val_std = sc_std.transform(X_tr), sc_std.transform(X_val)
    X_tr_mm, X_val_mm = sc_mm.transform(X_tr), sc_mm.transform(X_val)

    # Grilles reprend l'esprit Phase 4.
    grid = {
        "gaussian_var_smoothing": [1e-11, 1e-9, 1e-8, 1e-7],
        "mn_alpha": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
        "cn_alpha": [0.03, 0.05, 0.1, 0.2, 0.5, 1.0],
        "cn_norm": [False, True],
        "lr_C": [0.2, 0.5, 1.0, 2.0, 5.0],
        "lr_class_weight": [None, "balanced"],
    }

    best_g_acc, best_g_vs = -1.0, 1e-8
    for vs in grid["gaussian_var_smoothing"]:
        g = GaussianNB(var_smoothing=vs).fit(X_tr_std, y_tr)
        acc = accuracy_score(y_val, g.predict(X_val_std))
        if acc > best_g_acc:
            best_g_acc, best_g_vs = acc, vs

    best_m_acc, best_m_alpha = -1.0, 2.0
    for a in grid["mn_alpha"]:
        m = MultinomialNB(alpha=a).fit(X_tr_mm, y_tr)
        acc = accuracy_score(y_val, m.predict(X_val_mm))
        if acc > best_m_acc:
            best_m_acc, best_m_alpha = acc, a

    best_c_acc, best_c_alpha, best_c_norm = -1.0, 0.1, False
    for a in grid["cn_alpha"]:
        for nrm in grid["cn_norm"]:
            c = ComplementNB(alpha=a, norm=nrm).fit(X_tr_mm, y_tr)
            acc = accuracy_score(y_val, c.predict(X_val_mm))
            if acc > best_c_acc:
                best_c_acc, best_c_alpha, best_c_norm = acc, a, nrm

    best_l_acc, best_l_C, best_l_cw = -1.0, 1.0, None
    for c_val in grid["lr_C"]:
        for cw in grid["lr_class_weight"]:
            l = LogisticRegression(C=c_val, class_weight=cw, max_iter=2000, random_state=42).fit(X_tr_std, y_tr)
            acc = accuracy_score(y_val, l.predict(X_val_std))
            if acc > best_l_acc:
                best_l_acc, best_l_C, best_l_cw = acc, c_val, cw

    g_seed = GaussianNB(var_smoothing=best_g_vs).fit(X_tr_std, y_tr)
    m_seed = MultinomialNB(alpha=best_m_alpha).fit(X_tr_mm, y_tr)
    c_seed = ComplementNB(alpha=best_c_alpha, norm=best_c_norm).fit(X_tr_mm, y_tr)
    l_seed = LogisticRegression(C=best_l_C, class_weight=best_l_cw, max_iter=2000, random_state=42).fit(X_tr_std, y_tr)

    pg_val = g_seed.predict_proba(X_val_std)
    pm_val = m_seed.predict_proba(X_val_mm)
    pc_val = c_seed.predict_proba(X_val_mm)
    pl_val = l_seed.predict_proba(X_val_std)

    def weighted_proba(pg, pm, pc, pl, w):
        w1, w2, w3, w4 = w
        return (w1 * pg + w2 * pm + w3 * pc + w4 * pl) / (w1 + w2 + w3 + w4)

    seed_w = (0.2, 2.0, 1.5, 1.0)
    candidates = []
    for w in seed_w:
        options = sorted(set([
            0.2, 0.5, 1.0, 1.5, 2.0, 3.0,
            round(max(0.1, w - 0.4), 2),
            round(max(0.1, w - 0.2), 2),
            round(w, 2),
            round(w + 0.2, 2),
            round(w + 0.4, 2),
        ]))
        candidates.append(options)

    best_val_w_acc, best_w = -1.0, seed_w
    for w in itertools.product(*candidates):
        p = weighted_proba(pg_val, pm_val, pc_val, pl_val, w)
        y_hat = np.argmax(p, axis=1)
        acc = accuracy_score(y_val, y_hat)
        if acc > best_val_w_acc:
            best_val_w_acc, best_w = acc, w

    p_val_base = weighted_proba(pg_val, pm_val, pc_val, pl_val, best_w)
    y_val_base = np.argmax(p_val_base, axis=1)
    recall_by_class = recall_score(y_val, y_val_base, average=None, zero_division=0)
    weak_idx = np.argsort(recall_by_class)[:4].tolist()
    weak_idx = [i for i in weak_idx if recall_by_class[i] < 0.75]
    best_multipliers: dict[int, float] = {}
    best_thr_acc = best_val_w_acc

    if len(weak_idx) > 0:
        mult_grid = [1.00, 1.03, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20]
        best_combo = None
        for combo in itertools.product(mult_grid, repeat=len(weak_idx)):
            p_adj = p_val_base.copy()
            for j, cls_idx in enumerate(weak_idx):
                p_adj[:, cls_idx] *= combo[j]
            y_adj = np.argmax(p_adj, axis=1)
            acc = accuracy_score(y_val, y_adj)
            if acc > best_thr_acc:
                best_thr_acc = acc
                best_combo = combo
        if best_combo is not None:
            best_multipliers = {int(cls_idx): float(mult) for cls_idx, mult in zip(weak_idx, best_combo)}

    # Refit final sur train complet.
    g_full = GaussianNB(var_smoothing=best_g_vs).fit(X_train_std, y_train)
    m_full = MultinomialNB(alpha=best_m_alpha).fit(X_train_mm, y_train)
    c_full = ComplementNB(alpha=best_c_alpha, norm=best_c_norm).fit(X_train_mm, y_train)
    l_full = LogisticRegression(C=best_l_C, class_weight=best_l_cw, max_iter=2000, random_state=42).fit(X_train_std, y_train)

    pg_test = g_full.predict_proba(X_test_std)
    pm_test = m_full.predict_proba(X_test_mm)
    pc_test = c_full.predict_proba(X_test_mm)
    pl_test = l_full.predict_proba(X_test_std)
    p_test = weighted_proba(pg_test, pm_test, pc_test, pl_test, best_w)
    if len(best_multipliers) > 0:
        for cls_idx, mult in best_multipliers.items():
            p_test[:, cls_idx] *= mult
    y_pred_ensemble = np.argmax(p_test, axis=1)

    models = {
        "GaussianNB": g_full,
        "MultinomialNB": m_full,
        "ComplementNB": c_full,
        "LogisticRegression": l_full,
    }
    model_preprocessing = {
        "GaussianNB": "standard",
        "MultinomialNB": "minmax",
        "ComplementNB": "minmax",
        "LogisticRegression": "standard",
    }

    all_models = {}
    all_model_metrics = {}
    classification_reports = {}
    confusion_matrices = {}
    y_preds = {}
    y_probas = {}

    for model_name, model_obj in models.items():
        prep = model_preprocessing[model_name]
        if prep == "standard":
            model_obj.fit(X_train_std, y_train)
            y_pred = model_obj.predict(X_test_std)
            y_proba = model_obj.predict_proba(X_test_std)
        elif prep == "minmax":
            model_obj.fit(X_train_mm, y_train)
            y_pred = model_obj.predict(X_test_mm)
            y_proba = model_obj.predict_proba(X_test_mm)
        else:
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            y_proba = model_obj.predict_proba(X_test)

        all_models[model_name] = model_obj
        y_preds[model_name] = y_pred
        y_probas[model_name] = y_proba
        all_model_metrics[model_name] = _metric_dict(y_test, y_pred)
        classification_reports[model_name] = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)

    all_model_metrics[ensemble_name] = _metric_dict(y_test, y_pred_ensemble)
    classification_reports[ensemble_name] = classification_report(
        y_test,
        y_pred_ensemble,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    confusion_matrices[ensemble_name] = confusion_matrix(y_test, y_pred_ensemble)
    y_preds[ensemble_name] = y_pred_ensemble

    best_model_name = ensemble_name
    best_model = l_full
    best_scaler = scaler_standard

    feature_corr_df = X_df[all_features].corr().fillna(0.0)
    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "best_scaler": best_scaler,
        "scaler_standard": scaler_standard,
        "scaler_minmax": scaler_minmax,
        "model_preprocessing": model_preprocessing,
        "label_encoder": label_encoder,
        "all_features": all_features,
        "numeric_features": numeric_features,
        "encoded_features": encoded_features,
        "genre_features": genre_features,
        "company_features": company_features,
        "country_features": country_features,
        "le_lang": le_lang,
        "le_month": le_month,
        "le_season": le_season,
        "le_day": le_day,
        "le_homepage": le_homepage,
        "top_genres": top_genres,
        "top_companies": top_companies,
        "top_countries": top_countries,
        "top_languages": top_languages,
        "all_model_metrics": all_model_metrics,
        "classification_reports": classification_reports,
        "confusion_matrices": confusion_matrices,
        "feature_correlation": feature_corr_df.values,
        "feature_correlation_columns": feature_corr_df.columns.tolist(),
        "y_test": y_test,
        "y_preds": y_preds,
        "test_size": int(len(y_test)),
        "train_size": int(len(y_train)),
        "n_classes": int(len(label_encoder.classes_)),
        "n_features": int(len(all_features)),
        "all_models": all_models,
        "ensemble_model_name": ensemble_name,
        "ensemble_weights": tuple(float(x) for x in best_w),
        "ensemble_multipliers": {int(k): float(v) for k, v in best_multipliers.items()},
        "canonical_pipeline": "notebook_phase4b_like",
        "phase4b_val_acc_weights": float(best_val_w_acc),
        "phase4b_val_acc_thresholds": float(best_thr_acc),
        "phase4b_params": {
            "g_vs": float(best_g_vs),
            "mn_alpha": float(best_m_alpha),
            "cn_alpha": float(best_c_alpha),
            "cn_norm": bool(best_c_norm),
            "lr_C": float(best_l_C),
            "lr_cw": best_l_cw,
        },
    }


def make_unique_labels(labels):
    """Rend les labels uniques pour l'affichage (ex: has_collection, has_collection_2)."""
    counts = {}
    unique = []
    for label in labels:
        if label not in counts:
            counts[label] = 1
            unique.append(label)
        else:
            counts[label] += 1
            unique.append(f"{label}_{counts[label]}")
    return unique


def build_correlation_pairs(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Construit la liste des paires de features avec leur corrélation."""
    values = corr_df.values
    row_idx, col_idx = np.triu_indices_from(values, k=1)
    pairs = pd.DataFrame({
        "Feature A": corr_df.index[row_idx],
        "Feature B": corr_df.columns[col_idx],
        "Correlation": values[row_idx, col_idx],
    })
    pairs["AbsCorrelation"] = pairs["Correlation"].abs()
    return pairs.sort_values("AbsCorrelation", ascending=False)


def ensemble_predict_proba(model_probas: dict[str, np.ndarray], ensemble_weights, ensemble_multipliers) -> np.ndarray | None:
    """Agrège les probas selon la logique notebook Phase 4B (poids + multiplicateurs de classes)."""
    required = ["GaussianNB", "MultinomialNB", "ComplementNB", "LogisticRegression"]
    if any(name not in model_probas for name in required):
        return None

    if not ensemble_weights or len(ensemble_weights) != 4:
        ensemble_weights = (0.2, 2.0, 1.5, 1.0)

    w1, w2, w3, w4 = [float(x) for x in ensemble_weights]
    p = (
        w1 * model_probas["GaussianNB"]
        + w2 * model_probas["MultinomialNB"]
        + w3 * model_probas["ComplementNB"]
        + w4 * model_probas["LogisticRegression"]
    ) / (w1 + w2 + w3 + w4)

    if isinstance(ensemble_multipliers, dict):
        for cls_idx, mult in ensemble_multipliers.items():
            p[:, int(cls_idx)] *= float(mult)

    return p


try:
    if not os.path.exists(ARTIFACT_PATH):
        runtime_artifacts = build_runtime_artifacts_19(min_votes=50, n_target_classes=19)
        runtime_artifacts["artifact_source"] = "generated_from_notebook_pipeline"
        runtime_artifacts["artifact_generation_note"] = (
            "Artefact canonique absent: generation automatique depuis le pipeline notebook."
        )
        joblib.dump(runtime_artifacts, ARTIFACT_PATH)

    artifacts_mtime = os.path.getmtime(ARTIFACT_PATH)
    artifacts = load_artifacts(ARTIFACT_PATH, artifacts_mtime)
    best_model = artifacts["best_model"]
    best_model_name = artifacts["best_model_name"]
    best_scaler = artifacts.get("best_scaler")
    le = artifacts["label_encoder"]
    ALL_FEATURES = artifacts["all_features"]
    numeric_features = artifacts["numeric_features"]
    encoded_features = artifacts["encoded_features"]
    genre_features = artifacts["genre_features"]
    # Certaines anciennes versions d'artefacts incluent has_collection dans genre_features.
    # On l'exclut ici pour éviter d'écraser la vraie feature has_collection saisie par l'utilisateur.
    genre_feature_columns = [gf for gf in genre_features if gf != "has_collection"]
    company_features = artifacts["company_features"]
    country_features = artifacts["country_features"]
    le_lang = artifacts["le_lang"]
    le_month = artifacts["le_month"]
    le_season = artifacts["le_season"]
    le_day = artifacts["le_day"]
    le_homepage = artifacts["le_homepage"]
    top_genres = artifacts["top_genres"]
    top_companies = artifacts["top_companies"]
    top_countries = artifacts["top_countries"]
    top_languages = artifacts["top_languages"]
    all_model_metrics = artifacts["all_model_metrics"]
    classification_reports = artifacts["classification_reports"]
    confusion_mats = artifacts["confusion_matrices"]
    feature_corr = np.array(artifacts.get("feature_correlation", []))
    feature_corr_cols = artifacts.get("feature_correlation_columns", [])
    y_test_saved = np.array(artifacts["y_test"])
    y_preds = {k: np.array(v) for k, v in artifacts["y_preds"].items()}
    test_size = artifacts["test_size"]
    train_size = artifacts["train_size"]
    n_classes = artifacts["n_classes"]
    n_features = artifacts["n_features"]
    all_models = artifacts.get("all_models", {})
    model_preprocessing = artifacts.get("model_preprocessing", {})
    scaler_standard = artifacts.get("scaler_standard", best_scaler)
    scaler_minmax = artifacts.get("scaler_minmax", None)
    artifact_source = artifacts.get("artifact_source", "artifact_file")
    artifact_generation_note = artifacts.get("artifact_generation_note", "")
    canonical_pipeline = artifacts.get("canonical_pipeline", "")
    ensemble_model_name = artifacts.get("ensemble_model_name")
    ensemble_weights = artifacts.get("ensemble_weights", (0.2, 2.0, 1.5, 1.0))
    ensemble_multipliers = artifacts.get("ensemble_multipliers", {})

    # Exclure explicitement RandomForest de l'interface
    def _is_allowed_model_name(name: str) -> bool:
        return "randomforest" not in str(name).lower()

    all_model_metrics = {k: v for k, v in all_model_metrics.items() if _is_allowed_model_name(k)}
    classification_reports = {k: v for k, v in classification_reports.items() if _is_allowed_model_name(k)}
    confusion_mats = {k: v for k, v in confusion_mats.items() if _is_allowed_model_name(k)}
    y_preds = {k: v for k, v in y_preds.items() if _is_allowed_model_name(k)}
    all_models = {k: v for k, v in all_models.items() if _is_allowed_model_name(k)}

    # Si l'artefact a un meilleur modèle non autorisé, on bascule vers le meilleur modèle autorisé.
    if not _is_allowed_model_name(best_model_name):
        if len(all_model_metrics) == 0:
            raise ValueError("Aucun modèle autorisé disponible (RandomForest exclu).")
        best_model_name = max(all_model_metrics, key=lambda n: all_model_metrics[n]["accuracy"])
        if best_model_name in all_models:
            best_model = all_models[best_model_name]
        else:
            st.warning(
                "Le modèle best_model de l'artefact est exclu (RandomForest) et "
                "le modèle de remplacement n'est pas présent dans all_models. "
                "La prédiction utilisera le modèle chargé par défaut."
            )

    def transform_for_model(X: np.ndarray, model_name: str, model_obj):
        prep = model_preprocessing.get(model_name)
        if prep == "standard" and scaler_standard is not None:
            return scaler_standard.transform(X)
        if prep == "minmax" and scaler_minmax is not None:
            return scaler_minmax.transform(X)
        uses_nb_math = hasattr(model_obj, "feature_log_prob_") or hasattr(model_obj, "theta_")
        return best_scaler.transform(X) if (uses_nb_math and best_scaler is not None) else X

    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Erreur de chargement du modèle : {e}")
    st.info(
        "Générez un artefact modèle puis relancez l'app "
        "(par défaut `model_artifacts_notebook_canonical.pkl`)."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 Navigation")
page = st.sidebar.radio(
    "Choisir une page :",
    ["🔍 Exploration des données", "🎯 Prédiction", "📊 Analyse du modèle"],
)

st.sidebar.markdown("---")
if MODEL_LOADED:
    st.sidebar.markdown(
        f"""
**Projet ECE Paris**  
Apprentissage & Estimation Bayésienne  
*Prédiction du genre des films IMDb*  

**Modèle** : {best_model_name}  
**Artefact** : `{ARTIFACT_PATH}`  
**Features** : {n_features}  
**Accuracy** : {all_model_metrics[best_model_name]['accuracy']:.1%}
"""
    )
    st.sidebar.caption(f"Classes du modèle chargé ({len(le.classes_)}): {', '.join(le.classes_)}")
    if artifact_source == "generated_from_notebook_pipeline":
        st.sidebar.success("Pipeline notebook canonique actif")
        st.sidebar.caption(artifact_generation_note)
    if canonical_pipeline:
        st.sidebar.caption(f"Mode pipeline: {canonical_pipeline}")
    if ensemble_model_name:
        st.sidebar.caption(f"Ensemble: {ensemble_model_name} | poids={tuple(np.round(np.array(ensemble_weights, dtype=float), 2))}")
    if len(le.classes_) < 10:
        st.sidebar.warning(
            "Le modèle chargé est un modèle restreint (peu de genres). "
            "Le dataset complet peut contenir plus de genres, mais la prédiction restera bornée aux classes du modèle."
        )
    st.sidebar.info(
        "Mode avance: l'app affiche aussi une lecture Top-K (Top-3/Top-5) "
        "pour mieux refleter les cas multi-genres."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : Exploration des données
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Exploration des données":
    st.title("🔍 Exploration du Dataset IMDb")

    # Dataset complet transformé (notebook-like) + vue limitée au modèle
    df_all = load_dataset(tuple(), min_votes=50)
    if MODEL_LOADED:
        df_model = df_all[df_all["genre_target"].isin(le.classes_)].copy()
    else:
        df_model = df_all.copy()

    view_mode = st.radio(
        "Vue d'exploration :",
        ["Dataset complet transformé", "Sous-ensemble compatible modèle"],
        horizontal=True,
    )
    df_raw = df_all if view_mode == "Dataset complet transformé" else df_model

    st.markdown(
        f"**{len(df_raw):,} films** chargés depuis le dataset IMDb "
        "(version transformée alignée avec le pipeline du notebook)."
    )

    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Films", f"{len(df_raw):,}")
    col2.metric("Colonnes", df_raw.shape[1])
    col3.metric("Genres distincts", df_raw["genre_target"].nunique())
    col4.metric("Langues", df_raw["original_language"].nunique())

    if MODEL_LOADED:
        st.caption(
            f"Jeu d'evaluation du modele charge: {train_size + test_size:,} films | "
            f"Dataset complet transformé: {len(df_all):,} films | "
            f"Sous-ensemble classes modèle: {len(df_model):,} films."
        )

    st.markdown("---")

    # Aperçu du dataset
    with st.expander("📋 Aperçu du dataset (20 premières lignes)", expanded=False):
        cols_preview = [
            c for c in [
                "title", "year", "genre_target", "n_genres", "vote_average", "vote_count",
                "popularity", "runtime", "original_language", "has_collection"
            ] if c in df_raw.columns
        ]
        st.dataframe(df_raw[cols_preview].head(20), use_container_width=True)

    # Distribution des genres
    st.subheader("Distribution des genres")
    genre_counts = df_raw["genre_target"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Nombre"]

    fig_genre = px.bar(
        genre_counts,
        x="Genre",
        y="Nombre",
        color="Nombre",
        color_continuous_scale="Viridis",
        title="Distribution des genres (premier genre de chaque film)",
    )
    fig_genre.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_genre, use_container_width=True)

    # Distributions des features numériques
    st.subheader("Distributions des features numériques")
    numeric_feat = st.selectbox(
        "Choisir une feature :",
        ["vote_average", "vote_count", "popularity", "runtime", "budget", "revenue", "year"],
    )
    fig_hist = px.histogram(
        df_raw[df_raw[numeric_feat] > 0] if numeric_feat in ["budget", "revenue"] else df_raw,
        x=numeric_feat,
        nbins=50,
        title=f"Distribution de {numeric_feat}",
        marginal="box",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Corrélations lisibles entre features pertinentes
    st.subheader("Corrélations entre features pertinentes")
    if feature_corr.size > 0 and len(feature_corr_cols) > 0:
        display_cols = make_unique_labels(feature_corr_cols)
        corr_df = pd.DataFrame(feature_corr, index=display_cols, columns=display_cols)

        # 1) Heatmap focus: top features les plus corrélées en moyenne
        abs_corr = corr_df.abs().copy()
        np.fill_diagonal(abs_corr.values, 0.0)
        relevance = abs_corr.mean(axis=1).sort_values(ascending=False)

        top_k = st.slider(
            "Nombre de features à afficher dans la heatmap focus",
            min_value=8,
            max_value=min(35, len(corr_df)),
            value=min(18, len(corr_df)),
            step=1,
        )
        top_features = relevance.head(top_k).index.tolist()
        focused_corr = corr_df.loc[top_features, top_features]

        fig_corr_focus = px.imshow(
            focused_corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=f"Heatmap focus ({top_k} features les plus pertinentes)",
            aspect="auto",
            text_auto=".2f",
        )
        fig_corr_focus.update_layout(height=max(500, 24 * top_k))
        st.plotly_chart(fig_corr_focus, use_container_width=True)

        # 2) Tableau des paires les plus corrélées
        corr_pairs = build_correlation_pairs(corr_df)
        threshold = st.slider(
            "Seuil |corrélation| pour afficher les paires",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
        )
        filtered_pairs = corr_pairs[corr_pairs["AbsCorrelation"] >= threshold].copy()

        st.markdown(
            f"**{len(filtered_pairs)} paires** avec |corr| >= **{threshold:.2f}** "
            f"(sur {len(corr_pairs)} paires au total)."
        )
        st.dataframe(
            filtered_pairs.head(40).style.format({
                "Correlation": "{:.3f}",
                "AbsCorrelation": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # 3) Top corrélations positives / négatives
        top_pos = corr_pairs.sort_values("Correlation", ascending=False).head(10).copy()
        top_neg = corr_pairs.sort_values("Correlation", ascending=True).head(10).copy()
        top_pos["Pair"] = top_pos["Feature A"] + " ↔ " + top_pos["Feature B"]
        top_neg["Pair"] = top_neg["Feature A"] + " ↔ " + top_neg["Feature B"]

        col_pos, col_neg = st.columns(2)
        with col_pos:
            fig_pos = px.bar(
                top_pos.sort_values("Correlation"),
                x="Correlation",
                y="Pair",
                orientation="h",
                title="Top corrélations positives",
            )
            st.plotly_chart(fig_pos, use_container_width=True)
        with col_neg:
            fig_neg = px.bar(
                top_neg.sort_values("Correlation"),
                x="Correlation",
                y="Pair",
                orientation="h",
                title="Top corrélations négatives",
            )
            st.plotly_chart(fig_neg, use_container_width=True)
    else:
        st.warning("Corrélations indisponibles. Réexécutez le notebook pour regénérer les artefacts.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 : Prédiction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prédiction":
    st.title("🎯 Prédiction du Genre d'un Film")

    if not MODEL_LOADED:
        st.error("Le modèle n'est pas chargé. Exécutez d'abord le notebook.")
        st.stop()

    st.markdown(
        f"Modèle utilisé : **{best_model_name}** ({n_features} features, "
        f"{n_classes} genres, accuracy {all_model_metrics[best_model_name]['accuracy']:.1%})"
    )
    st.markdown(
        "Renseignez les caractéristiques d'un film ci-dessous, puis cliquez sur **Prédire**."
    )

    # Bornes dynamiques pour garder une UI réaliste sur les très gros films.
    df_pred_ref = load_dataset(tuple(), min_votes=50)
    vote_count_ref_max = int(df_pred_ref["vote_count"].max()) if len(df_pred_ref) > 0 else 50000
    vote_count_ui_max = int(max(1_000_000, vote_count_ref_max * 2 + 1000))
    popularity_ref_max = float(df_pred_ref["popularity"].max()) if len(df_pred_ref) > 0 else 500.0
    popularity_ui_max = float(max(5000.0, popularity_ref_max * 2 + 10.0))

    # Formulaire de saisie
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Métriques")
        vote_count = st.number_input(
            "Nombre de votes",
            min_value=50,
            max_value=vote_count_ui_max,
            value=500,
            step=50,
            help=f"Borne max UI ajustée automatiquement ({vote_count_ui_max:,}) d'après le dataset.",
        )
        vote_average = st.slider("Note moyenne", 0.0, 10.0, 6.5, 0.1)
        popularity = st.number_input(
            "Score de popularité",
            min_value=0.0,
            max_value=popularity_ui_max,
            value=15.0,
            step=1.0,
            help=(
                f"Borne max UI ajustée automatiquement ({popularity_ui_max:.1f}) "
                "d'après le dataset."
            ),
        )
        runtime = st.slider("Durée (minutes)", 10, 300, 100, 5)
        year = st.slider("Année de sortie", 1950, 2025, 2020)

    with col2:
        st.markdown("#### 📅 Infos")
        language = st.selectbox("Langue originale", top_languages + ["Other"])
        month_options = list(le_month.classes_)
        month = st.selectbox("Mois de sortie", month_options)
        season_options = list(le_season.classes_)
        season = st.selectbox("Saison", season_options)
        day_options = list(le_day.classes_)
        day_of_week = st.selectbox("Jour de sortie", day_options)
        has_homepage = st.selectbox("Page web officielle", list(le_homepage.classes_))
        has_collection = st.checkbox("Fait partie d'une saga", value=False)

    st.markdown("---")

    # Section pour genres, companies et countries (en expanders pour ne pas surcharger)
    with st.expander("🎭 Genres associés (optionnel)"):
        st.caption(
            "Ces genres sont utilisés comme vecteur d'information complémentaire "
            "(mode binaire ou pondéré)."
        )
        genre_vector_mode = st.radio(
            "Mode d'encodage des genres :",
            options=["Binaire (multi-hot)", "Pondéré"],
            horizontal=True,
        )
        selected_genres = st.multiselect(
            "Sélectionnez les genres associés connus :",
            options=top_genres,
            default=[],
        )
        genre_weights = {}
        if genre_vector_mode == "Pondéré" and selected_genres:
            st.caption("Ajustez le poids de chaque genre (0 = ignoré, 1 = maximum).")
            default_weights = [1.0, 0.6, 0.3]
            for idx, genre_name in enumerate(selected_genres):
                default_weight = default_weights[idx] if idx < len(default_weights) else 0.2
                genre_weights[genre_name] = st.slider(
                    f"Poids — {genre_name}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_weight),
                    step=0.05,
                    key=f"weight_{genre_name}",
                )
        elif selected_genres:
            genre_weights = {genre_name: 1.0 for genre_name in selected_genres}

    with st.expander("🏢 Compagnies de production (optionnel)"):
        selected_companies = st.multiselect(
            "Sélectionnez les compagnies de production :",
            options=top_companies,
            default=[],
        )

    with st.expander("🌍 Pays de production (optionnel)"):
        selected_countries = st.multiselect(
            "Sélectionnez les pays de production :",
            options=top_countries,
            default=[],
        )

    st.markdown("")

    if st.button("🎯 Prédire le genre", type="primary", use_container_width=True):
        # Construire le vecteur de features
        feature_values = {}

        # Numériques
        feature_values["vote_count"] = vote_count
        feature_values["vote_average"] = vote_average
        feature_values["popularity"] = popularity
        feature_values["runtime"] = runtime
        feature_values["year"] = year

        # Encodées
        feature_values["lang_enc"] = le_lang.transform([language])[0] if language in le_lang.classes_ else 0
        feature_values["month_enc"] = le_month.transform([month])[0]
        feature_values["season_enc"] = le_season.transform([season])[0]
        feature_values["day_enc"] = le_day.transform([day_of_week])[0]
        feature_values["homepage_enc"] = le_homepage.transform([has_homepage])[0]
        feature_values["has_collection"] = int(has_collection)

        # Genres associés (multi-hot binaire ou pondéré)
        for gf in genre_feature_columns:
            genre_name = gf.removeprefix("has_")
            if genre_name in selected_genres:
                if genre_vector_mode == "Pondéré":
                    feature_values[gf] = float(genre_weights.get(genre_name, 0.0))
                else:
                    feature_values[gf] = 1.0
            else:
                feature_values[gf] = 0.0

        # Company multi-hot
        for i, cf in enumerate(company_features):
            feature_values[cf] = 1 if (i < len(top_companies) and top_companies[i] in selected_companies) else 0

        # Country multi-hot
        for i, ctf in enumerate(country_features):
            feature_values[ctf] = 1 if (i < len(top_countries) and top_countries[i] in selected_countries) else 0

        # DataFrame
        X_input = pd.DataFrame([feature_values])[ALL_FEATURES].values.astype(np.float64)
        X_model_input = transform_for_model(X_input, best_model_name, best_model)

        # Probabilités par modèle autorisé puis agrégation de consensus.
        models_for_cmp = all_models if len(all_models) > 0 else {best_model_name: best_model}
        per_model_proba = {}
        per_model_pred = {}
        for model_name, model_obj in models_for_cmp.items():
            if "randomforest" in model_name.lower():
                continue
            X_cmp = transform_for_model(X_input, model_name, model_obj)
            try:
                proba = model_obj.predict_proba(X_cmp)[0]
                pred_idx = int(np.argmax(proba))
                per_model_proba[model_name] = proba
                per_model_pred[model_name] = pred_idx
            except Exception:
                continue

        ensemble_proba = ensemble_predict_proba(
            per_model_proba,
            ensemble_weights=ensemble_weights,
            ensemble_multipliers=ensemble_multipliers,
        )

        if ensemble_proba is not None:
            probabilities_raw = ensemble_proba[0]
            pred_source = ensemble_model_name or "Ensemble notebook"
        else:
            best_model_proba = per_model_proba.get(best_model_name)
            if best_model_proba is None:
                best_model_proba = best_model.predict_proba(X_model_input)[0]
            probabilities_raw = best_model_proba
            pred_source = best_model_name

        prediction = int(np.argmax(probabilities_raw))
        probabilities = probabilities_raw / max(float(np.sum(probabilities_raw)), 1e-12)
        predicted_genre = le.classes_[prediction]

        # Affichage
        st.success(f"### Genre prédit : **{predicted_genre}**")
        st.caption(f"Source de prédiction: {pred_source} (logique notebook canonique).")

        # Barplot des probabilités
        prob_df = pd.DataFrame({
            "Genre": le.classes_,
            "Probabilité": probabilities,
        }).sort_values("Probabilité", ascending=True)

        fig_prob = px.bar(
            prob_df,
            x="Probabilité",
            y="Genre",
            orientation="h",
            color="Probabilité",
            color_continuous_scale="Blues",
            title=f"Probabilités par genre ({pred_source})",
        )
        fig_prob.update_layout(height=600, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_prob, use_container_width=True)

        # Top 5 genres
        top5 = prob_df.nlargest(5, "Probabilité")
        st.markdown("**Top 5 genres les plus probables :**")
        for _, row in top5.iterrows():
            pct = row["Probabilité"] * 100
            st.write(f"- **{row['Genre']}** : {pct:.2f}%")

        # Comparaison des prédictions entre modèles autorisés
        st.markdown("### Comparaison des modèles (même entrée)")
        model_cmp_rows = []
        for model_name, model_obj in models_for_cmp.items():
            if "randomforest" in model_name.lower():
                continue

            X_cmp = transform_for_model(X_input, model_name, model_obj)

            try:
                if model_name in per_model_pred and model_name in per_model_proba:
                    pred_idx = int(per_model_pred[model_name])
                    proba = per_model_proba[model_name]
                else:
                    pred_idx = int(model_obj.predict(X_cmp)[0])
                    proba = model_obj.predict_proba(X_cmp)[0]
                pred_name = le.classes_[pred_idx]
                top1_prob = float(np.max(proba))
                top3_idx = np.argsort(proba)[::-1][:3]
                top3_names = [le.classes_[i] for i in top3_idx]
                model_cmp_rows.append({
                    "Modèle": model_name,
                    "Top-1 prédit": pred_name,
                    "Confiance Top-1": top1_prob,
                    "Top-3": " | ".join(top3_names),
                })
            except Exception as model_exc:
                model_cmp_rows.append({
                    "Modèle": model_name,
                    "Top-1 prédit": f"Erreur: {model_exc}",
                    "Confiance Top-1": np.nan,
                    "Top-3": "-",
                })

        if len(model_cmp_rows) > 0:
            cmp_df = pd.DataFrame(model_cmp_rows).sort_values("Confiance Top-1", ascending=False, na_position="last")
            st.dataframe(
                cmp_df.style.format({"Confiance Top-1": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Aucun modèle de comparaison disponible dans l'artefact chargé.")

        # Lecture Top-K (utile si le film est multi-genre)
        st.markdown("### Lecture multi-genres (Top-K)")
        ranked_idx = np.argsort(probabilities)[::-1]
        top1_idx = int(ranked_idx[0])
        top3_idx = ranked_idx[:3]
        top5_idx = ranked_idx[:5]

        top1_name = le.classes_[top1_idx]
        top1_prob = float(probabilities[top1_idx])
        top2_prob = float(probabilities[int(ranked_idx[1])]) if len(ranked_idx) > 1 else 0.0
        margin = top1_prob - top2_prob

        c_top1, c_margin, c_k = st.columns(3)
        c_top1.metric("Top-1", top1_name, f"{top1_prob*100:.1f}%")
        c_margin.metric("Marge Top-1 vs Top-2", f"{margin*100:.1f} pts")
        c_k.metric("Top-3 cumule", f"{probabilities[top3_idx].sum()*100:.1f}%")

        top3_labels = [le.classes_[i] for i in top3_idx]
        top5_labels = [le.classes_[i] for i in top5_idx]

        st.write(f"**Top-3 recommande** : {', '.join(top3_labels)}")
        with st.expander("Afficher le Top-5 detaille"):
            for rank, class_idx in enumerate(top5_idx, start=1):
                st.write(f"{rank}. **{le.classes_[class_idx]}** — {probabilities[class_idx]*100:.2f}%")

        if margin < 0.08:
            st.warning(
                "Prediction ambigue: le Top-1 est proche du Top-2. "
                "Interpretez plutot le resultat comme un Top-3 genres probables."
            )
        else:
            st.success("Prediction relativement stable (marge Top-1 suffisante).")

        # Pondération / contribution des features au score du genre prédit
        st.markdown("### Pondération des features dans le calcul du modèle")
        if hasattr(best_model, "feature_log_prob_") and hasattr(best_model, "class_log_prior_"):
            x_scaled = X_model_input[0]
            class_idx = int(prediction)
            class_name = le.classes_[class_idx]
            weights = best_model.feature_log_prob_[class_idx]
            contributions = x_scaled * weights

            contrib_df = pd.DataFrame({
                "Feature": ALL_FEATURES,
                "Valeur (après scaling)": x_scaled,
                "Poids log P(feature|classe)": weights,
                "Contribution au score": contributions,
            })
            contrib_df["AbsContribution"] = contrib_df["Contribution au score"].abs()
            contrib_df = contrib_df.sort_values("AbsContribution", ascending=False)

            st.caption(
                f"Classe analysée : {class_name}. "
                "Contribution = valeur_feature_scaled × log P(feature|classe)."
            )
            st.dataframe(
                contrib_df[["Feature", "Valeur (après scaling)", "Poids log P(feature|classe)", "Contribution au score"]]
                .head(20)
                .style.format({
                    "Valeur (après scaling)": "{:.4f}",
                    "Poids log P(feature|classe)": "{:.4f}",
                    "Contribution au score": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            fig_contrib = px.bar(
                contrib_df.head(20).sort_values("Contribution au score"),
                x="Contribution au score",
                y="Feature",
                orientation="h",
                title="Top 20 contributions de features (classe prédite)",
                color="Contribution au score",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
        elif hasattr(best_model, "theta_") and hasattr(best_model, "var_"):
            x_scaled = X_model_input[0]
            class_idx = int(prediction)
            class_name = le.classes_[class_idx]
            mu = best_model.theta_[class_idx]
            var = np.maximum(best_model.var_[class_idx], 1e-12)
            contributions = -0.5 * np.log(2 * np.pi * var) - ((x_scaled - mu) ** 2) / (2 * var)

            contrib_df = pd.DataFrame({
                "Feature": ALL_FEATURES,
                "Valeur (après scaling)": x_scaled,
                "Moyenne classe (theta)": mu,
                "Variance classe": var,
                "Contribution log-vraisemblance": contributions,
            })
            contrib_df["AbsContribution"] = contrib_df["Contribution log-vraisemblance"].abs()
            contrib_df = contrib_df.sort_values("AbsContribution", ascending=False)

            st.caption(
                f"Classe analysée : {class_name}. "
                "Contribution = terme log-vraisemblance Gaussienne par feature."
            )
            st.dataframe(
                contrib_df[[
                    "Feature",
                    "Valeur (après scaling)",
                    "Moyenne classe (theta)",
                    "Variance classe",
                    "Contribution log-vraisemblance",
                ]]
                .head(20)
                .style.format({
                    "Valeur (après scaling)": "{:.4f}",
                    "Moyenne classe (theta)": "{:.4f}",
                    "Variance classe": "{:.4f}",
                    "Contribution log-vraisemblance": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            fig_contrib = px.bar(
                contrib_df.head(20).sort_values("Contribution log-vraisemblance"),
                x="Contribution log-vraisemblance",
                y="Feature",
                orientation="h",
                title="Top 20 contributions de features (classe prédite)",
                color="Contribution log-vraisemblance",
                color_continuous_scale="RdBu_r",
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            st.info("Le modèle courant n'expose pas de pondérations de features interprétables.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 : Analyse du modèle
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse du modèle":
    st.title("📊 Analyse des Performances du Modèle")

    if not MODEL_LOADED:
        st.error("Le modèle n'est pas chargé. Exécutez d'abord le notebook.")
        st.stop()

    # ── En-tête : infos du modèle ────────────────────────────────────────────
    st.markdown(f"**Meilleur modèle** : `{best_model_name}` — **{n_features} features**, **{n_classes} genres**")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("Features", n_features)
    col_info2.metric("Classes", n_classes)
    col_info3.metric("Train", f"{train_size:,}")
    col_info4.metric("Test", f"{test_size:,}")

    st.markdown(
        f"*{len(numeric_features)} numériques, "
        f"{len(encoded_features)} catégorielles encodées, "
        f"{len(genre_features)} genres (multi-hot), "
        f"{len(company_features)} compagnies (multi-hot), "
        f"{len(country_features)} pays (multi-hot)*"
    )

    st.markdown("---")

    # Options de modele reutilisees par plusieurs sections
    _model_options = list(classification_reports.keys())
    _default_model_index = _model_options.index(best_model_name) if best_model_name in _model_options else 0

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1B : Diagnostic "Pourquoi pas 90% ?"
    # ══════════════════════════════════════════════════════════════════════════
    st.header("1B. Diagnostic: pourquoi le score plafonne")

    selected_diag_model = st.selectbox(
        "Modèle à diagnostiquer :",
        _model_options,
        index=_default_model_index,
        key="diag_model",
    )
    diag_report = classification_reports[selected_diag_model]
    diag_metrics = all_model_metrics[selected_diag_model]
    diag_cm = np.array(confusion_mats[selected_diag_model])

    # Build per-genre metrics table
    diag_rows = []
    for genre_name in le.classes_:
        if genre_name in diag_report:
            gr = diag_report[genre_name]
            diag_rows.append({
                "Genre": genre_name,
                "Precision": float(gr.get("precision", 0.0)),
                "Recall": float(gr.get("recall", 0.0)),
                "F1": float(gr.get("f1-score", 0.0)),
                "Support": int(gr.get("support", 0)),
            })

    diag_df = pd.DataFrame(diag_rows)
    weak_df = diag_df.sort_values("Recall").head(5).copy()

    # Top confusions (hors diagonale)
    cm_work = diag_cm.copy().astype(float)
    np.fill_diagonal(cm_work, 0.0)
    ridx, cidx = np.where(cm_work > 0)
    conf_rows = []
    for r, c in zip(ridx, cidx):
        conf_rows.append({
            "Reel": le.classes_[int(r)],
            "Predit": le.classes_[int(c)],
            "Erreurs": int(cm_work[r, c]),
        })
    conf_df = pd.DataFrame(conf_rows).sort_values("Erreurs", ascending=False) if len(conf_rows) > 0 else pd.DataFrame(columns=["Reel", "Predit", "Erreurs"])

    d1, d2, d3 = st.columns(3)
    d1.metric("Accuracy", f"{diag_metrics['accuracy']:.2%}")
    d2.metric("F1 macro", f"{diag_metrics['f1_macro']:.4f}")
    d3.metric("F1 weighted", f"{diag_metrics['f1_weighted']:.4f}")

    macro_weighted_gap = float(diag_metrics["f1_weighted"] - diag_metrics["f1_macro"])
    if macro_weighted_gap > 0.06:
        st.info(
            "Gap F1 weighted - F1 macro eleve: le modele est globalement bon, "
            "mais certaines classes restent sensiblement plus difficiles."
        )

    c_weak, c_conf = st.columns(2)
    with c_weak:
        st.markdown("**Genres les plus difficiles (Recall le plus bas)**")
        st.dataframe(
            weak_df.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with c_conf:
        st.markdown("**Confusions les plus frequentes**")
        st.dataframe(conf_df.head(10), use_container_width=True, hide_index=True)

    st.caption(
        "Lecture conseillee: si les memes genres dominent dans les erreurs, "
        "le plafond vient plutot du recouvrement des classes et des features disponibles "
        "que d'un simple manque de tuning."
    )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 : Métriques globales
    # ══════════════════════════════════════════════════════════════════════════
    st.header("1. Métriques globales")

    # KPIs du meilleur modèle
    best_m = all_model_metrics[best_model_name]
    st.subheader(f"{best_model_name} — Résumé")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Accuracy", f"{best_m['accuracy']:.2%}")
    k2.metric("F1 weighted", f"{best_m['f1_weighted']:.4f}")
    k3.metric("F1 macro", f"{best_m['f1_macro']:.4f}")
    k4.metric("Precision weighted", f"{best_m['precision_weighted']:.4f}")
    k5.metric("Recall weighted", f"{best_m['recall_weighted']:.4f}")
    k6.metric("Taux d'erreur", f"{1 - best_m['accuracy']:.2%}")

    st.markdown("")

    # Tableau comparatif
    st.subheader("Comparaison des modèles disponibles")
    comp_rows = []
    for model_name, metrics in all_model_metrics.items():
        comp_rows.append({
            "Modèle": model_name,
            "Accuracy": f"{metrics['accuracy']:.2%}",
            "F1 weighted": f"{metrics['f1_weighted']:.4f}",
            "F1 macro": f"{metrics['f1_macro']:.4f}",
            "Precision weighted": f"{metrics['precision_weighted']:.4f}",
            "Precision macro": f"{metrics['precision_macro']:.4f}",
            "Recall weighted": f"{metrics['recall_weighted']:.4f}",
            "Recall macro": f"{metrics['recall_macro']:.4f}",
        })
    comp_table = pd.DataFrame(comp_rows)
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    # Graphique comparatif
    comp_chart_data = []
    for model_name, metrics in all_model_metrics.items():
        for metric_name, value in metrics.items():
            comp_chart_data.append({
                "Modèle": model_name,
                "Métrique": metric_name,
                "Score": value,
            })
    comp_chart_df = pd.DataFrame(comp_chart_data)

    metric_choice = st.multiselect(
        "Métriques à afficher :",
        options=list(all_model_metrics[list(all_model_metrics.keys())[0]].keys()),
        default=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
    )
    if metric_choice:
        filtered = comp_chart_df[comp_chart_df["Métrique"].isin(metric_choice)]
        fig_comp = px.bar(
            filtered,
            x="Modèle",
            y="Score",
            color="Métrique",
            barmode="group",
            title="Comparaison des modèles disponibles",
            text_auto=".3f",
        )
        fig_comp.update_layout(yaxis_range=[0, 1], height=500)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 : Classification report par genre
    # ══════════════════════════════════════════════════════════════════════════
    st.header("2. Performance par genre")

    selected_report_model = st.selectbox(
        "Modèle :", _model_options, index=_default_model_index, key="report_model",
    )
    report = classification_reports[selected_report_model]

    report_rows = []
    for genre in le.classes_:
        if genre in report:
            r = report[genre]
            report_rows.append({
                "Genre": genre,
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1-Score": r["f1-score"],
                "Support": int(r["support"]),
            })

    report_df = pd.DataFrame(report_rows)

    st.dataframe(
        report_df.style.format({
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "Support": "{:d}",
        }).background_gradient(subset=["Precision", "Recall", "F1-Score"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    fig_genre_perf = go.Figure()
    fig_genre_perf.add_trace(go.Bar(name="Precision", x=report_df["Genre"], y=report_df["Precision"]))
    fig_genre_perf.add_trace(go.Bar(name="Recall", x=report_df["Genre"], y=report_df["Recall"]))
    fig_genre_perf.add_trace(go.Bar(name="F1-Score", x=report_df["Genre"], y=report_df["F1-Score"]))
    fig_genre_perf.update_layout(
        barmode="group",
        title=f"Precision / Recall / F1 par genre — {selected_report_model}",
        yaxis_range=[0, 1],
        xaxis_tickangle=-45,
        height=500,
    )
    st.plotly_chart(fig_genre_perf, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 : Matrices de confusion
    # ══════════════════════════════════════════════════════════════════════════
    st.header("3. Matrices de confusion")

    _cm_options = list(confusion_mats.keys())
    selected_cm_model = st.selectbox(
        "Modèle :", _cm_options,
        index=_cm_options.index(best_model_name) if best_model_name in _cm_options else 0,
        key="cm_model",
    )
    cm = np.array(confusion_mats[selected_cm_model])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig_cm = px.imshow(
        cm_norm,
        x=list(le.classes_),
        y=list(le.classes_),
        color_continuous_scale="Blues",
        title=f"Matrice de confusion normalisée — {selected_cm_model}",
        text_auto=".2f",
        aspect="auto",
        labels=dict(x="Prédit", y="Réel", color="Recall"),
    )
    fig_cm.update_layout(height=700)
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("Matrice de confusion (valeurs absolues)"):
        fig_cm_abs = px.imshow(
            cm,
            x=list(le.classes_),
            y=list(le.classes_),
            color_continuous_scale="Oranges",
            title=f"Matrice de confusion (absolue) — {selected_cm_model}",
            text_auto="d",
            aspect="auto",
            labels=dict(x="Prédit", y="Réel", color="Nb films"),
        )
        fig_cm_abs.update_layout(height=700)
        st.plotly_chart(fig_cm_abs, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 : Distribution réelle vs prédite
    # ══════════════════════════════════════════════════════════════════════════
    st.header("4. Distribution réelle vs prédite")

    _dist_options = list(y_preds.keys())
    selected_dist_model = st.selectbox(
        "Modèle :", _dist_options,
        index=_dist_options.index(best_model_name) if best_model_name in _dist_options else 0,
        key="dist_model",
    )
    y_pred_sel = y_preds[selected_dist_model]

    real_labels = [le.classes_[i] for i in y_test_saved]
    pred_labels = [le.classes_[i] for i in y_pred_sel]

    real_counts = pd.Series(real_labels).value_counts().reindex(le.classes_, fill_value=0)
    pred_counts = pd.Series(pred_labels).value_counts().reindex(le.classes_, fill_value=0)

    dist_df = pd.DataFrame({
        "Genre": list(le.classes_) * 2,
        "Nombre": list(real_counts.values) + list(pred_counts.values),
        "Type": ["Réel"] * n_classes + ["Prédit"] * n_classes,
    })
    fig_dist = px.bar(
        dist_df, x="Genre", y="Nombre", color="Type", barmode="group",
        title=f"Distribution réelle vs prédite — {selected_dist_model}",
    )
    fig_dist.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 : Détail des features
    # ══════════════════════════════════════════════════════════════════════════
    st.header("5. Détail des features utilisées")

    feat_breakdown = pd.DataFrame({
        "Catégorie": ["Numériques", "Catégorielles encodées", "Genres (multi-hot)",
                      "Compagnies (multi-hot)", "Pays (multi-hot)"],
        "Nombre": [len(numeric_features), len(encoded_features), len(genre_features),
                   len(company_features), len(country_features)],
        "Exemples": [
            ", ".join(numeric_features),
            ", ".join(encoded_features),
            ", ".join(genre_features[:5]) + "...",
            ", ".join(company_features[:3]) + "...",
            ", ".join(country_features[:3]) + "...",
        ],
    })
    st.dataframe(feat_breakdown, use_container_width=True, hide_index=True)

    fig_feat = px.pie(
        feat_breakdown,
        values="Nombre",
        names="Catégorie",
        title="Répartition des types de features",
        hole=0.4,
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    with st.expander("Liste complète des features"):
        for i, f in enumerate(ALL_FEATURES, 1):
            st.text(f"{i:3d}. {f}")
