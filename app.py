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
APP_NAME = "Movie Finder"
LOGO_PATH = "logo.png"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des artefacts ──────────────────────────────────────────────────
DEFAULT_ARTIFACT_PATH = "model_artifacts.pkl"
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


def parse_list_column(value):
    """Parse une colonne stockée en liste sérialisée dans le CSV."""
    if value in ("not available", "[]") or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


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
        "(par défaut `model_artifacts.pkl`)."
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=120)
st.sidebar.title(f"🎬 {APP_NAME}")
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
*{APP_NAME} — Prédiction du genre des films IMDb*  

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
    st.title("🔍 Exploration des 4 genres")

    df_raw = load_dataset()
    st.caption("Vue synthétique pour la présentation: Action, Comedy, Drama, Horror.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Films", f"{len(df_raw):,}")
    col2.metric("Genres", "4")
    col3.metric("Note moyenne", f"{df_raw['vote_average'].mean():.2f}")
    col4.metric("Durée médiane", f"{df_raw['runtime'].median():.0f} min")

    genre_counts = df_raw["first_genre"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Nombre"]
    fig_genre = px.bar(
        genre_counts,
        x="Genre",
        y="Nombre",
        color="Genre",
        title="Répartition des films par genre",
    )
    st.plotly_chart(fig_genre, use_container_width=True)

    fig_box = px.box(
        df_raw,
        x="first_genre",
        y="vote_average",
        color="first_genre",
        title="Distribution des notes par genre",
    )
    st.plotly_chart(fig_box, use_container_width=True)


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

    input_mode = st.radio(
        "Mode de saisie :",
        options=["Saisie manuelle", "Film du dataset"],
        index=1,
        horizontal=True,
    )
    dataset_mode = input_mode == "Film du dataset"
    selected_movie_row = None

    month_options = list(le_month.classes_)
    season_options = list(le_season.classes_)
    day_options = list(le_day.classes_)
    homepage_options = list(le_homepage.classes_)

    vote_count_default = 500
    vote_average_default = 6.5
    popularity_default = 15.0
    runtime_default = 100
    year_default = 2020
    language_default = top_languages[0] if top_languages else "Other"
    month_default = month_options[0] if month_options else "Jan"
    season_default = season_options[0] if season_options else "Q1"
    day_default = day_options[0] if day_options else "Friday"
    homepage_default = homepage_options[0] if homepage_options else "NO"
    has_collection_default = False
    selected_genres_default = []
    selected_companies_default = []
    selected_countries_default = []

    if dataset_mode:
        df_pred = load_dataset().reset_index(drop=True)
        movie_idx = st.selectbox(
            "Choisissez un film du dataset :",
            options=df_pred.index.tolist(),
            format_func=lambda idx: (
                f"{df_pred.at[idx, 'title']} ({int(df_pred.at[idx, 'year']) if pd.notna(df_pred.at[idx, 'year']) else 'N/A'})"
                f" — genre réel: {df_pred.at[idx, 'first_genre']}"
            ),
        )
        selected_movie_row = df_pred.loc[movie_idx]
        st.caption("Les champs ci-dessous sont préremplis automatiquement à partir du film sélectionné.")

        vote_count_default = int(max(50, float(selected_movie_row.get("vote_count", 500) or 500)))
        vote_average_default = float(np.clip(float(selected_movie_row.get("vote_average", 6.5) or 6.5), 0.0, 10.0))
        popularity_default = float(max(0.0, float(selected_movie_row.get("popularity", 15.0) or 15.0)))
        runtime_default = int(np.clip(float(selected_movie_row.get("runtime", 100) or 100), 10, 300))
        year_default = int(np.clip(float(selected_movie_row.get("year", 2020) or 2020), 1950, 2025))

        raw_language = str(selected_movie_row.get("original_language", "Other"))
        language_default = raw_language if raw_language in (top_languages + ["Other"]) else "Other"

        raw_month = str(selected_movie_row.get("month", month_default))
        month_default = raw_month if raw_month in month_options else month_default
        raw_season = str(selected_movie_row.get("season", season_default))
        season_default = raw_season if raw_season in season_options else season_default
        raw_day = str(selected_movie_row.get("day_of_week", day_default))
        day_default = raw_day if raw_day in day_options else day_default

        raw_homepage = str(selected_movie_row.get("has_homepage", homepage_default))
        homepage_default = raw_homepage if raw_homepage in homepage_options else homepage_default
        has_collection_default = str(selected_movie_row.get("belongs_to_collection", "not available")) not in [
            "not available", "", "[]",
        ]

        selected_genres_default = [g for g in parse_list_column(selected_movie_row.get("genre")) if g in top_genres]
        selected_companies_default = [
            c for c in parse_list_column(selected_movie_row.get("companies")) if c in top_companies
        ]
        selected_countries_default = [
            c for c in parse_list_column(selected_movie_row.get("countries")) if c in top_countries
        ]

    if dataset_mode:
        # Mode direct: aucune saisie manuelle, on prend exactement les valeurs du film sélectionné.
        vote_count = vote_count_default
        vote_average = vote_average_default
        popularity = popularity_default
        runtime = runtime_default
        year = year_default
        language = language_default
        month = month_default
        season = season_default
        day_of_week = day_default
        has_homepage = homepage_default
        has_collection = has_collection_default
        genre_vector_mode = "Binaire (multi-hot)"
        selected_genres = selected_genres_default
        selected_companies = selected_companies_default
        selected_countries = selected_countries_default
        genre_weights = {genre_name: 1.0 for genre_name in selected_genres}

        st.info(
            f"Film sélectionné: **{selected_movie_row['title']}** ({int(selected_movie_row['year'])}) "
            f"— genre réel: **{selected_movie_row['first_genre']}**"
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Votes", f"{int(vote_count)}")
        m2.metric("Note", f"{vote_average:.1f}/10")
        m3.metric("Popularité", f"{popularity:.1f}")
        m4.metric("Durée", f"{int(runtime)} min")
        m5.metric("Année", f"{int(year)}")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Métriques")
            vote_count = st.number_input(
                "Nombre de votes", min_value=50, max_value=50000, value=vote_count_default, step=50,
            )
            vote_average = st.slider("Note moyenne", 0.0, 10.0, vote_average_default, 0.1)
            popularity = st.number_input(
                "Score de popularité", min_value=0.0, max_value=500.0, value=popularity_default, step=1.0,
            )
            runtime = st.slider("Durée (minutes)", 10, 300, runtime_default, 5)
            year = st.slider("Année de sortie", 1950, 2025, year_default)

        with col2:
            st.markdown("#### 📅 Infos")
            language = st.selectbox(
                "Langue originale", top_languages + ["Other"],
                index=(top_languages + ["Other"]).index(language_default),
            )
            month = st.selectbox("Mois de sortie", month_options, index=month_options.index(month_default))
            season = st.selectbox("Saison", season_options, index=season_options.index(season_default))
            day_of_week = st.selectbox("Jour de sortie", day_options, index=day_options.index(day_default))
            has_homepage = st.selectbox(
                "Page web officielle", homepage_options, index=homepage_options.index(homepage_default),
            )
            has_collection = st.checkbox("Fait partie d'une saga", value=has_collection_default)

        st.markdown("---")

        with st.expander("🎭 Genres associés (optionnel)"):
            genre_vector_mode = st.radio(
                "Mode d'encodage des genres :",
                options=["Binaire (multi-hot)", "Pondéré"],
                horizontal=True,
            )
            selected_genres = st.multiselect(
                "Genres associés (4 genres affichés):",
                options=list(le.classes_),
                default=[g for g in selected_genres_default if g in set(le.classes_)],
            )
            genre_weights = {}
            if genre_vector_mode == "Pondéré" and selected_genres:
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
                default=selected_companies_default,
            )

        with st.expander("🌍 Pays de production (optionnel)"):
            selected_countries = st.multiselect(
                "Sélectionnez les pays de production :",
                options=top_countries,
                default=selected_countries_default,
            )

    st.markdown("")

    if st.button("🎯 Prédire le genre", type="primary", use_container_width=True):
        # Construire le vecteur de features
        feature_values = {}

        # En mode dataset, on force les métriques à partir du film choisi
        # pour éviter toute divergence avec l'affichage.
        if dataset_mode and selected_movie_row is not None:
            vote_count = int(max(50, float(selected_movie_row.get("vote_count", vote_count_default) or vote_count_default)))
            vote_average = float(np.clip(float(selected_movie_row.get("vote_average", vote_average_default) or vote_average_default), 0.0, 10.0))
            popularity = float(max(0.0, float(selected_movie_row.get("popularity", popularity_default) or popularity_default)))
            runtime = int(np.clip(float(selected_movie_row.get("runtime", runtime_default) or runtime_default), 10, 300))
            year = int(np.clip(float(selected_movie_row.get("year", year_default) or year_default), 1950, 2025))

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
        if selected_movie_row is not None:
            true_genre = str(selected_movie_row.get("first_genre", "Inconnu"))
            is_correct = predicted_genre == true_genre
            if is_correct:
                st.success(f"✅ Vérification dataset : prédiction correcte (réel = **{true_genre}**).")
            else:
                st.warning(f"❌ Vérification dataset : réel = **{true_genre}**, prédit = **{predicted_genre}**.")

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

        # Top 4 genres (les 4 classes du modèle)
        top4 = prob_df.nlargest(4, "Probabilité")
        st.markdown("**Probabilités (4 genres) :**")
        for _, row in top4.iterrows():
            pct = row["Probabilité"] * 100
            st.write(f"- **{row['Genre']}** : {pct:.2f}%")

        # Pondération / contribution des features au score du genre prédit
        with st.expander("Détails techniques (contributions)", expanded=False):
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
