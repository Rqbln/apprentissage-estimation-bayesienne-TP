"""
Streamlit App — Prédiction du Genre des Films avec Naive Bayes
==============================================================
Interface graphique interactive pour explorer le dataset, prédire le genre
d'un film et analyser les performances du modèle.

Modèle : ComplementNB avec 82 features (multi-hot encoding)
Accuracy : ~70%

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

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Prédiction Genre Films — Naive Bayes",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Chargement des artefacts ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Charge le modèle entraîné et les métadonnées."""
    return joblib.load("model_artifacts.pkl")


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
    return df


try:
    artifacts = load_artifacts()
    best_model = artifacts["best_model"]
    best_model_name = artifacts["best_model_name"]
    best_scaler = artifacts["best_scaler"]
    le = artifacts["label_encoder"]
    ALL_FEATURES = artifacts["all_features"]
    numeric_features = artifacts["numeric_features"]
    encoded_features = artifacts["encoded_features"]
    genre_features = artifacts["genre_features"]
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
    st.info("Exécutez d'abord le notebook pour générer `model_artifacts.pkl`.")

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

    # Corrélations
    st.subheader("Matrice de corrélation")
    numeric_cols = ["budget", "popularity", "revenue", "runtime", "vote_average", "vote_count", "year"]
    corr = df_raw[numeric_cols].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Corrélations entre features numériques",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


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
    col1, col2, col3 = st.columns(3)

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

    with col3:
        st.markdown("#### 🎭 Genres secondaires du film")
        st.caption("Cochez les genres qui décrivent le film (en plus du genre principal qu'on cherche à prédire).")
        selected_genres = []
        # Show in 2 sub-columns
        g_col1, g_col2 = st.columns(2)
        for i, genre in enumerate(top_genres):
            target_col = g_col1 if i < len(top_genres) // 2 else g_col2
            if target_col.checkbox(genre, key=f"genre_{genre}"):
                selected_genres.append(genre)

    st.markdown("---")

    # Section pour companies et countries (en expanders pour ne pas surcharger)
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

        # Genre multi-hot
        for gf in genre_features:
            genre_name = gf.replace("has_", "")
            feature_values[gf] = 1 if genre_name in selected_genres else 0

        # Company multi-hot
        for i, cf in enumerate(company_features):
            feature_values[cf] = 1 if (i < len(top_companies) and top_companies[i] in selected_companies) else 0

        # Country multi-hot
        for i, ctf in enumerate(country_features):
            feature_values[ctf] = 1 if (i < len(top_countries) and top_countries[i] in selected_countries) else 0

        # DataFrame
        X_input = pd.DataFrame([feature_values])[ALL_FEATURES].values.astype(np.float64)

        # Scaling
        X_input_scaled = best_scaler.transform(X_input)

        # Prédiction
        prediction = best_model.predict(X_input_scaled)[0]
        probabilities = best_model.predict_proba(X_input_scaled)[0]
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
    st.subheader("Comparaison des variantes Naive Bayes")
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
            title="Comparaison des variantes Naive Bayes",
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
        "Modèle :", list(classification_reports.keys()), key="report_model",
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

    # Ajouter les moyennes
    for avg_key in ["macro avg", "weighted avg"]:
        if avg_key in report:
            r = report[avg_key]
            report_rows.append({
                "Genre": avg_key.upper(),
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

    # Graphique par genre (sans les moyennes)
    report_df_genres = report_df[~report_df["Genre"].str.contains("AVG|MACRO|WEIGHTED", case=False, na=False)]

    fig_genre_perf = go.Figure()
    fig_genre_perf.add_trace(go.Bar(name="Precision", x=report_df_genres["Genre"], y=report_df_genres["Precision"]))
    fig_genre_perf.add_trace(go.Bar(name="Recall", x=report_df_genres["Genre"], y=report_df_genres["Recall"]))
    fig_genre_perf.add_trace(go.Bar(name="F1-Score", x=report_df_genres["Genre"], y=report_df_genres["F1-Score"]))
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

    selected_cm_model = st.selectbox(
        "Modèle :", list(confusion_mats.keys()), key="cm_model",
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

    selected_dist_model = st.selectbox(
        "Modèle :", list(y_preds.keys()), key="dist_model",
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
