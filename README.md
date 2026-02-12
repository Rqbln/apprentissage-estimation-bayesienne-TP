# Prédiction du genre des films — Naive Bayes

Projet ECE Paris — **Apprentissage & Estimation Bayésienne**  
Classification du genre de films IMDb avec des modèles Naive Bayes (GaussianNB, MultinomialNB, ComplementNB).

## Données

- **Dataset** : [IMDb Movies (user-friendly)](https://www.kaggle.com/datasets/jacopoferretti/idmb-movies-user-friendly) via `kagglehub`
- **Cible** : genre principal du film (19 classes : Action, Comedy, Drama, etc.)
- **Features** : 82 variables (métriques numériques, multi-hot encoding genres/compagnies/pays, infos temporelles et langue)

## Modèle

- **Meilleur modèle** : **ComplementNB** (~70 % accuracy sur le jeu de test)
- Comparaison avec GaussianNB et MultinomialNB dans le notebook et dans l’app Streamlit

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Le dataset est téléchargé automatiquement au premier lancement (notebook ou Streamlit).

## Utilisation

### Lancer l’application Streamlit

```bash
streamlit run app.py
```

- **Exploration** : aperçu du dataset, distributions, corrélations  
- **Prédiction** : formulaire interactif pour prédire le genre d’un film  
- **Analyse** : métriques, matrices de confusion, performance par genre  

### Réentraîner le modèle (Jupyter)

1. Ouvrir `Projet_Prediction_Genre_Films_NaiveBayes.ipynb`
2. Exécuter toutes les cellules
3. Les artefacts sont sauvegardés dans `model_artifacts.pkl` (utilisé par l’app Streamlit)

## Structure du dépôt

| Fichier | Description |
|--------|-------------|
| `Projet_Prediction_Genre_Films_NaiveBayes.ipynb` | Pipeline complet : chargement, nettoyage, feature engineering, entraînement, sauvegarde |
| `app.py` | Interface Streamlit (exploration, prédiction, analyse) |
| `model_artifacts.pkl` | Modèle entraîné, scaler, encodeurs, métriques (généré par le notebook) |
| `requirements.txt` | Dépendances Python |

## Auteurs

Projet réalisé dans le cadre du cours Apprentissage & Estimation Bayésienne — ECE Paris.
