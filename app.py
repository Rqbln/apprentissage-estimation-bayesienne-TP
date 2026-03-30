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

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Genre Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des artefacts ──────────────────────────────────────────────────
DEFAULT_ARTIFACT_PATH = (
    "model_artifacts_over88.pkl"
    if os.path.exists("model_artifacts_over88.pkl")
    else "model_artifacts.pkl"
)
ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH)


@st.cache_resource
def load_artifacts(artifact_path: str, artifacts_mtime: float):
    """Charge le modèle entraîné et les métadonnées."""
    return joblib.load(artifact_path)


@st.cache_data
def load_dataset():
    """Charge et prépare le dataset."""
    import kagglehub
    path = kagglehub.dataset_download("jacopoferretti/idmb-movies-user-friendly")
    df = pd.read_csv(f"{path}/MOVIES.csv")

    def extract_first_genre(x):
        if x == "not available" or x == "[]":
            return None
        try:
            lst = ast.literal_eval(x)
            return lst[0] if len(lst) > 0 else None
        except Exception:
            return None

    df["first_genre"] = df["genre"].apply(extract_first_genre)
    allowed_genres = ["Drama", "Comedy", "Action", "Horror"]
    df = df[df["first_genre"].isin(allowed_genres)].copy()
    return df


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


try:
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
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Erreur de chargement du modèle : {e}")
    st.info(
        "Générez un artefact modèle puis relancez l'app "
        "(par défaut `model_artifacts_over88.pkl`, sinon `model_artifacts.pkl`)."
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 : Exploration des données
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Exploration des données":
    st.title("🔍 Exploration du Dataset IMDb")

    df_raw = load_dataset()

    st.markdown(f"**{len(df_raw):,} films** chargés depuis le dataset IMDb.")

    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Films", f"{len(df_raw):,}")
    col2.metric("Colonnes", df_raw.shape[1])
    col3.metric("Genres distincts", df_raw["first_genre"].nunique())
    col4.metric("Langues", df_raw["original_language"].nunique())

    st.markdown("---")

    # Aperçu du dataset
    with st.expander("📋 Aperçu du dataset (20 premières lignes)", expanded=False):
        st.dataframe(df_raw.head(20), use_container_width=True)

    # Distribution des genres
    st.subheader("Distribution des genres")
    genre_counts = df_raw["first_genre"].value_counts().reset_index()
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

    # Formulaire de saisie
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Métriques")
        vote_count = st.number_input("Nombre de votes", min_value=50, max_value=50000, value=500, step=50)
        vote_average = st.slider("Note moyenne", 0.0, 10.0, 6.5, 0.1)
        popularity = st.number_input("Score de popularité", min_value=0.0, max_value=500.0, value=15.0, step=1.0)
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

        # Transformation pour le modèle courant:
        # - Naive Bayes: conserve le scaling
        # - autres modèles (ex. RandomForest): pas de scaling
        uses_nb_math = hasattr(best_model, "feature_log_prob_") or hasattr(best_model, "theta_")
        X_model_input = best_scaler.transform(X_input) if (uses_nb_math and best_scaler is not None) else X_input

        # Prédiction
        prediction = best_model.predict(X_model_input)[0]
        probabilities = best_model.predict_proba(X_model_input)[0]
        predicted_genre = le.classes_[prediction]

        # Affichage
        st.success(f"### Genre prédit : **{predicted_genre}**")

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
            title=f"Probabilités par genre ({best_model_name})",
        )
        fig_prob.update_layout(height=600, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_prob, use_container_width=True)

        # Top 5 genres
        top5 = prob_df.nlargest(5, "Probabilité")
        st.markdown("**Top 5 genres les plus probables :**")
        for _, row in top5.iterrows():
            pct = row["Probabilité"] * 100
            st.write(f"- **{row['Genre']}** : {pct:.2f}%")

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

    # Liste des modèles (meilleur modèle par défaut)
    _model_options = list(classification_reports.keys())
    _default_model_index = _model_options.index(best_model_name) if best_model_name in _model_options else 0

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
