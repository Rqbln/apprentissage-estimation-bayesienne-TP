# Documentation du modèle de prédiction de genre

Ce document décrit précisément comment le modèle est entraîné, quelles informations d'un film sont analysées, et comment les prédictions sont produites dans l'application Streamlit.

## 1) Objectif du modèle

Prédire le **genre principal** d'un film parmi 4 classes:

- `Action`
- `Comedy`
- `Drama`
- `Horror`

Le modèle est multi-classes et renvoie:
- une classe prédite
- une distribution de probabilités sur les 4 genres

## 2) Source de données

Dataset utilisé:
- Kaggle: `jacopoferretti/idmb-movies-user-friendly`
- Chargé via `kagglehub` dans le notebook `Projet_Prediction_Genre_Films_NaiveBayes.ipynb`

Colonnes principales exploitées:
- numériques: `vote_count`, `vote_average`, `popularity`, `runtime`, `year`
- contextuelles/catégorielles: `month`, `season`, `day_of_week`, `original_language`, `has_homepage`
- structure film: `belongs_to_collection`, `companies`, `countries`, `genre`

## 3) Nettoyage et filtrage

Avant entraînement, le pipeline applique:

1. Parsing des colonnes-listes (`genre`, `companies`, `countries`, `belongs_to_collection`)
2. Extraction du **genre principal** (`genre_target`) = premier genre de la liste
3. Filtre des films sans genre principal
4. Filtre des genres conservés: seulement `Action`, `Comedy`, `Drama`, `Horror`
5. Filtre qualité: `vote_count >= 50`
6. Imputation `runtime` manquant par médiane intra-genre
7. Encodage de `has_homepage`

## 4) Ce qui est analysé pour chaque film (features)

Chaque film est transformé en **82 features**:

- 5 features numériques:
  - `vote_count`
  - `vote_average`
  - `popularity`
  - `runtime`
  - `year`

- 6 features catégorielles encodées:
  - `lang_enc` (langue)
  - `month_enc`
  - `season_enc`
  - `day_enc`
  - `homepage_enc`
  - `has_collection`

- 21 features liées aux genres (`has_*`) dont:
  - `has_collection` (feature structure film)
  - 20 indicateurs de genres associés (`has_<genre>`)
- 30 features multi-hot de compagnies (`company_*`)
- 20 features multi-hot de pays (`country_*`)

Remarques:
- L'UI de prédiction permet désormais de renseigner les genres associés avec 2 modes:
  - **Binaire (multi-hot)**: présence/absence par genre
  - **Pondéré**: poids continu entre 0 et 1 par genre
- `has_collection` reste pilotée uniquement par la case à cocher "Fait partie d'une saga" (elle n'est plus écrasée par le bloc genres).

## 5) Prétraitement pour l'entraînement

- Encodage de la cible avec `LabelEncoder`
- Encodage des variables catégorielles avec `LabelEncoder`
- Split train/test stratifié:
  - `test_size = 0.2`
  - `random_state = 42`
- Scalers testés:
  - `StandardScaler` pour `GaussianNB`
  - `MinMaxScaler` pour `MultinomialNB` et `ComplementNB`

## 6) Modèles entraînés et sélection du meilleur

Trois variantes Naive Bayes sont entraînées:

- `GaussianNB`
- `MultinomialNB`
- `ComplementNB`

Le meilleur modèle est choisi par **accuracy test**.

État actuel des artefacts (`model_artifacts.pkl`):

- Meilleur modèle: `GaussianNB`
- Classes: `['Action', 'Comedy', 'Drama', 'Horror']`
- Taille train: `4951`
- Taille test: `1238`
- Features: `82`

## 7) Métriques

Les métriques sauvegardées pour chaque modèle:

- accuracy
- f1 weighted / macro
- precision weighted / macro
- recall weighted / macro
- classification report complet
- matrice de confusion

Exemple (meilleur modèle actuel, extrait des artefacts):
- Accuracy: `0.8813`
- F1 weighted: `0.8831`
- F1 macro: `0.8733`

## 8) Artefacts sauvegardés

Le notebook sauvegarde `model_artifacts.pkl` avec:

- modèle(s) entraîné(s)
- scaler(s)
- encodeurs (`LabelEncoder`, `MultiLabelBinarizer`)
- liste de features
- métriques
- confusion matrices
- prédictions test
- matrice de corrélation des features (82x82)

Ces artefacts sont la source unique de vérité utilisée par `app.py`.

## 9) Interprétation de la prédiction dans Streamlit

Après prédiction, l'app affiche:

1. Genre prédit
2. Probabilités par genre
3. Contributions des features au score de la classe prédite:
   - pour `Multinomial/Complement`: contribution basée sur `feature_log_prob_`
   - pour `GaussianNB`: contribution de log-vraisemblance via `theta_` et `var_`

Cela permet d'expliquer **pourquoi** le modèle choisit un genre donné.

## 10) Évaluation vectorielle (roadmap implémentée)

Un script reproductible a été ajouté pour comparer les scénarios de migration vers des vecteurs de genres:

- Fichier: `genre_vector_experiments.py`
- Sorties générées:
  - `genre_vector_experiment_report.md`
  - `genre_vector_experiment_results.json`

### Protocole

- Dataset Kaggle identique (`jacopoferretti/idmb-movies-user-friendly`)
- Filtrage identique (`vote_count >= 50`, 4 genres cibles)
- Split stratifié (`random_state = 42`)
- KPI suivis: accuracy, F1 weighted, F1 macro, precision weighted, recall weighted

### Scénarios comparés

1. `baseline_ui_zero_genres`: simulation de l'ancienne UI (genres à 0)
2. `phase1_multihot_binary`: genres associés en binaire
3. `phase2_multihot_weighted`: genres associés pondérés (1.0, 0.6, 0.3, puis 0.2)

Les expériences sont exécutées sans fuite de cible: seuls les **genres secondaires** (`genre_list[1:]`) sont encodés dans les scénarios phase 1/phase 2.

### Gate embeddings (phase 3)

La décision de lancer une phase embeddings est pilotée par un seuil de gain (`F1 macro`) du mode pondéré par rapport au mode binaire.

- Si gain >= seuil: stabiliser phase 2, ne pas engager embeddings.
- Sinon: ouvrir un spike embeddings dans une branche dédiée.

## 11) Reproduire un entraînement complet

1. Activer l'environnement:

```bash
source venv/bin/activate
```

2. Réexécuter tout le notebook:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 "Projet_Prediction_Genre_Films_NaiveBayes.ipynb" --output "Projet_Prediction_Genre_Films_NaiveBayes.ipynb"
```

3. Vérifier que `model_artifacts.pkl` est régénéré.

## 12) Lancer les expériences vecteurs de genres

```bash
source venv/bin/activate
python genre_vector_experiments.py
```

Le script regénère:
- `genre_vector_experiment_report.md`
- `genre_vector_experiment_results.json`

