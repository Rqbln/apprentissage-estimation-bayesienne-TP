# Rapport d'expériences — Vecteurs de genres

## Baseline verrouillée (artefacts actuels)
- Accuracy: 0.8813
- F1 weighted: 0.8831
- F1 macro: 0.8733

## Résultats de migration

| Scénario | Modèle | Accuracy | F1 weighted | F1 macro |
|---|---|---:|---:|---:|
| `baseline_ui_zero_genres` | `GaussianNB` | 0.4459 | 0.4333 | 0.4004 |
| `phase1_multihot_binary` | `GaussianNB` | 0.3691 | 0.3728 | 0.3776 |
| `phase2_multihot_weighted` | `GaussianNB` | 0.4225 | 0.4324 | 0.4342 |
| `phase2_multihot_weighted` | `LogisticRegression(multinomial)` | 0.7302 | 0.7274 | 0.7021 |

## Décision de gate Phase 3 (embeddings)
- Seuil de décision (gain F1 macro pondéré vs phase 1): 0.0050
- Gain observé: 0.0566
- Décision: Le gain du vecteur pondéré est suffisant: priorité à la stabilisation Phase 2. La phase embeddings reste désactivée pour l'instant.

## Meilleur scénario observé
- `phase2_multihot_weighted` avec `LogisticRegression(multinomial)` (F1 macro = 0.7021)
