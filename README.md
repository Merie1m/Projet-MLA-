# Plateforme ML — Interface d'Entraînement et d'Évaluation de Modèles

Plateforme web complète pour l'entraînement, l'évaluation et le suivi de modèles de Machine Learning (classification et régression), avec intégration MLflow pour le tracking d'expériences.

---

## Table des matières

1. [Architecture du projet](#architecture-du-projet)
2. [Technologies utilisées](#technologies-utilisées)
3. [Installation et lancement](#installation-et-lancement)
4. [Étapes réalisées](#étapes-réalisées)
5. [Fonctionnalités principales](#fonctionnalités-principales)
6. [Endpoints de l'API backend](#endpoints-de-lapi-backend)
7. [Algorithmes supportés](#algorithmes-supportés)
8. [Remarques importantes](#remarques-importantes)

---

## Architecture du projet

```
projet_ml_front/
├── backend/                    # API FastAPI (Python)
│   ├── main.py                 # Routes API (18 endpoints)
│   ├── ml_engine.py            # Moteur ML (entraînement, évaluation, auto-tuning, PCA/t-SNE)
│   ├── store.py                # Stockage en mémoire (datasets, modèles, expériences)
│   ├── requirements.txt        # Dépendances Python
│   ├── models/                 # Modèles sauvegardés (.joblib)
│   └── mlruns/                 # Données MLflow (créé automatiquement)
├── src/                        # Frontend React
│   ├── App.jsx                 # Routeur principal
│   ├── constants.js            # Définitions algorithmes, métriques, stratégies
│   ├── main.jsx                # Point d'entrée React
│   ├── index.css               # Styles Tailwind
│   ├── components/             # Composants réutilisables
│   │   ├── Layout.jsx          # Layout avec sidebar navigation
│   │   ├── HyperparamForm.jsx  # Formulaire dynamique d'hyperparamètres
│   │   ├── StatCard.jsx        # Carte de statistique
│   │   └── HelpTooltip.jsx     # Tooltip d'aide
│   ├── pages/                  # Pages de l'application
│   │   ├── DashboardPage.jsx   # Tableau de bord général
│   │   ├── DatasetPage.jsx     # Upload, nettoyage, exploration, PCA/t-SNE
│   │   ├── TrainingPage.jsx    # Sélection d'algorithme et entraînement
│   │   ├── ResultsPage.jsx     # Comparaison des résultats et visualisations
│   │   ├── ModelsPage.jsx      # Gestion des modèles sauvegardés
│   │   └── MLOpsPage.jsx       # Suivi des expériences et MLflow
│   ├── services/
│   │   └── api.js              # Client HTTP (fetch vers le backend)
│   └── store/
│       └── useStore.js         # État global Zustand
├── index.html
├── vite.config.js              # Config Vite + proxy /api → localhost:8000
├── tailwind.config.js
├── postcss.config.js
└── package.json
```

---

## Technologies utilisées

### Frontend
| Technologie | Rôle |
|---|---|
| **React 18** | Bibliothèque UI |
| **Vite 5** | Bundler / serveur dev |
| **Material UI 5** | Composants UI (boutons, tableaux, dialogues, etc.) |
| **Tailwind CSS 3** | Utilitaires CSS |
| **Recharts 2** | Graphiques (bar, radar, scatter, line, etc.) |
| **Zustand 4** | Gestion d'état global |
| **React Router v6** | Navigation SPA |
| **PapaParse** | Parsing CSV côté client |
| **react-dropzone** | Zone de drag & drop pour upload |
| **react-hot-toast** | Notifications toast |
| **html-to-image + file-saver** | Export de graphiques en PNG |

### Backend
| Technologie | Rôle |
|---|---|
| **FastAPI** | Framework API REST |
| **Uvicorn** | Serveur ASGI |
| **scikit-learn** | Algorithmes ML (classification + régression) |
| **XGBoost** | Gradient Boosting |
| **Optuna** | Optimisation bayésienne des hyperparamètres |
| **MLflow** | Tracking d'expériences, logging de métriques et modèles |
| **Pandas / NumPy** | Manipulation de données |
| **Joblib** | Sérialisation des modèles |

---

## Installation et lancement

### Prérequis
- **Python 3.10+** (testé avec Python 3.13)
- **Node.js 18+** et npm

### 1. Cloner le projet
```bash
cd projet_ml_front
```

### 2. Installer les dépendances frontend
```bash
npm install
```

### 3. Créer l'environnement Python et installer les dépendances backend
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 4. Lancer le backend (port 8000)
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Lancer le frontend (port 3000)
```bash
# Dans un autre terminal, à la racine du projet
npm run dev -- --port 3000
```

### 6. Accéder à l'application
Ouvrir **http://localhost:3000** dans le navigateur.

> Le fichier `vite.config.js` contient un proxy qui redirige automatiquement les appels `/api/*` vers `http://localhost:8000`.

---

## Étapes réalisées

### Tâche 2 — Interface frontend complète

1. **Page Dataset** — Upload CSV/Excel via drag & drop, aperçu des données (tableau paginé), statistiques descriptives (types, valeurs manquantes, distribution), opérations de nettoyage (suppression NA, remplissage moyenne/médiane/mode, normalisation, standardisation, encodage label/one-hot), sélection de la colonne cible.

2. **Page Training** — Sélection d'algorithme parmi 8 algorithmes de classification, configuration dynamique des hyperparamètres, réglage du ratio train/test, lancement de l'entraînement avec suivi en temps réel (polling).

3. **Page Results** — Comparaison multi-modèles avec graphiques (bar chart des métriques, matrice de confusion interactive, courbe ROC, courbe Precision-Recall, radar chart, tableau récapitulatif exportable en CSV).

4. **Page Models** — Liste des modèles entraînés, détails et métriques, versioning, rollback, export au format `.joblib`.

5. **Page MLOps** — Historique des expériences, filtres par algorithme/date, détails avec métriques dans un dialogue modal.

6. **Page Dashboard** — Vue d'ensemble avec cartes de statistiques (nombre de datasets, modèles, meilleur modèle, expériences), graphique de performance.

7. **Backend FastAPI** — Création complète du backend avec stockage en mémoire, routes REST, moteur ML avec 8 algorithmes de classification, auto-tuning (Grid Search, Random Search, Optuna).

### Tâche 3 — MLflow, Régression et Réduction de dimension

1. **Intégration MLflow** :
   - Chaque entraînement crée un run MLflow automatiquement.
   - Logging des paramètres (algorithme, hyperparamètres, dataset, colonne cible, ratio train/test).
   - Logging des métriques (accuracy, precision, recall, F1, ROC AUC, log loss pour la classification ; MAE, MSE, RMSE, R² pour la régression).
   - Logging du modèle en artefact via `mlflow.sklearn.log_model()`.
   - Endpoint `/api/mlflow/status` pour vérifier l'état du tracking.
   - Endpoint `/api/mlflow/runs` pour lister les runs et permettre la comparaison.
   - Les données MLflow sont stockées dans `backend/mlruns/`.

2. **Algorithmes de régression** (3 ajoutés) :
   - **Régression Linéaire** — Modèle linéaire classique, sans hyperparamètres.
   - **SVR (Support Vector Regression)** — Avec noyaux RBF/poly/linear/sigmoid, paramètres C, gamma, epsilon.
   - **Random Forest Régression** — Ensemble d'arbres, paramètres n_estimators, max_depth, min_samples_split, min_samples_leaf.
   - Métriques de régression : MAE, MSE, RMSE, R².
   - Auto-tuning adapté pour la régression (scoring `neg_mean_squared_error`).

3. **Réduction de dimension (PCA / t-SNE)** :
   - Endpoint `POST /api/dimension-reduction` pour calculer PCA ou t-SNE.
   - Visualisation scatter plot interactif sur la page Dataset.
   - Basculement PCA ↔ t-SNE via toggle buttons.
   - Affichage de la variance expliquée pour PCA.
   - Échantillonnage automatique à 5 000 lignes max pour les performances.

4. **Adaptation de l'UI** :
   - La page Results affiche dynamiquement les onglets selon le type de problème (classification ou régression).
   - Onglet "Résidus" (scatter Actual vs Predicted) pour les modèles de régression.
   - La page Dashboard adapte la métrique du meilleur modèle (Accuracy pour classification, R² pour régression).
   - La page MLOps affiche le type de problème et le MLflow Run ID pour chaque expérience.
   - Le Training affiche R² ou Accuracy selon le type d'algorithme dans les notifications de succès.

---

## Fonctionnalités principales

| Fonctionnalité | Description |
|---|---|
| Upload de données | CSV et Excel, drag & drop, aperçu paginé |
| Nettoyage de données | 9 opérations (drop NA, fill mean/median/mode, drop duplicates, normalisation, standardisation, label encoding, one-hot) |
| Entraînement | 11 algorithmes (8 classification + 3 régression), configuration des hyperparamètres |
| Auto-tuning | Grid Search, Random Search, Optuna (bayésien) |
| Évaluation | Métriques classification (accuracy, precision, recall, F1, ROC AUC, log loss) + régression (MAE, MSE, RMSE, R²) |
| Visualisations | Bar chart, matrice de confusion, courbe ROC, courbe PR, radar, scatter résidus |
| MLflow tracking | Logging automatique des params, métriques et artefacts modèle |
| PCA / t-SNE | Réduction de dimension avec scatter plot coloré par classe |
| Export | Modèles en `.joblib`, résultats en CSV, graphiques en PNG |
| Versioning | Historique des versions de chaque modèle, rollback possible |

---

## Endpoints de l'API backend

| Méthode | Route | Description |
|---|---|---|
| `POST` | `/api/dataset/upload` | Upload un fichier CSV/Excel |
| `GET` | `/api/dataset/{id}/preview` | Aperçu paginé d'un dataset |
| `POST` | `/api/dataset/clean` | Appliquer des opérations de nettoyage |
| `GET` | `/api/models` | Lister les modèles entraînés |
| `GET` | `/api/models/{id}` | Détails d'un modèle |
| `GET` | `/api/models/{algo}/versions` | Versions d'un algorithme |
| `POST` | `/api/models/{id}/rollback` | Rollback vers une version précédente |
| `POST` | `/api/train` | Lancer un entraînement |
| `GET` | `/api/train/status/{job_id}` | État d'un entraînement en cours |
| `GET` | `/api/results` | Tous les résultats d'expériences |
| `GET` | `/api/results/{id}` | Détail d'un résultat |
| `POST` | `/api/autotune` | Lancer l'auto-tuning |
| `GET` | `/api/experiments` | Historique des expériences |
| `GET` | `/api/export/model/{id}` | Télécharger un modèle (.joblib) |
| `POST` | `/api/dimension-reduction` | Calculer PCA ou t-SNE |
| `GET` | `/api/mlflow/status` | État du tracking MLflow |
| `GET` | `/api/mlflow/runs` | Lister les runs MLflow |
| `GET` | `/api/health` | Vérification de santé de l'API |

---

## Algorithmes supportés

### Classification (8)

| Algorithme | ID | Hyperparamètres principaux |
|---|---|---|
| Support Vector Machine | `svm` | C, kernel, gamma, degree |
| Random Forest | `random_forest` | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| K-Nearest Neighbors | `knn` | n_neighbors, weights, metric, p |
| Régression Logistique | `logistic_regression` | C, penalty, solver, max_iter |
| Réseau de Neurones (MLP) | `neural_network` | hidden_layers, learning_rate, activation, optimizer, epochs, batch_size, dropout |
| Gradient Boosting (XGBoost) | `gradient_boosting` | n_estimators, learning_rate, max_depth, subsample, colsample_bytree |
| Arbre de Décision | `decision_tree` | max_depth, criterion, min_samples_split, min_samples_leaf |
| Naive Bayes | `naive_bayes` | var_smoothing |

### Régression (3)

| Algorithme | ID | Hyperparamètres principaux |
|---|---|---|
| Régression Linéaire | `linear_regression` | aucun |
| SVR | `svr` | C, kernel, gamma, epsilon |
| Random Forest Régression | `rf_regression` | n_estimators, max_depth, min_samples_split, min_samples_leaf |

---

## Remarques importantes

### Stockage en mémoire
Le backend utilise un **stockage en mémoire** (`store.py`). Toutes les données (datasets, modèles, expériences) sont perdues lorsque le serveur est redémarré. Seuls les fichiers de modèles exportés (.joblib dans `backend/models/`) et les données MLflow (`backend/mlruns/`) persistent sur disque.

### Encodage des colonnes catégorielles
Pour éviter une **explosion mémoire** lors du one-hot encoding de colonnes à haute cardinalité, le moteur utilise automatiquement le **Label Encoding** pour les colonnes avec plus de 10 valeurs uniques, et le One-Hot Encoding pour les colonnes avec 10 valeurs ou moins.

### Échantillonnage automatique
- **Entraînement** : les datasets de plus de **50 000 lignes** sont automatiquement échantillonnés.
- **PCA / t-SNE** : échantillonnage à **5 000 lignes** maximum pour des performances raisonnables.

### Stratification
La stratification du split train/test est appliquée automatiquement en classification, **sauf** si certaines classes ont moins de 2 exemples (dans ce cas, un split aléatoire simple est utilisé).

### MLflow
- MLflow est configuré pour stocker les données localement dans `backend/mlruns/`.
- Chaque entraînement crée automatiquement un run dans l'expérience `ML_Platform_Experiments`.
- Les runs MLflow sont consultables via l'interface (page MLOps) ou via la CLI MLflow : `mlflow ui --backend-store-uri backend/mlruns/`.

### Proxy Vite
Le fichier `vite.config.js` configure un proxy pour rediriger `/api/*` vers `http://localhost:8000`. En production, il faudrait configurer un reverse proxy (nginx, etc.) pour servir le frontend statique et router les appels API.

### Python 3.13
Le projet a été développé et testé avec **Python 3.13**. Certaines dépendances (pandas, numpy) nécessitent des versions récentes pour supports cette version de Python. Les version pins ont été retirés de `requirements.txt` pour assurer la compatibilité.

### Export des résultats
- Les **modèles** sont exportables en `.joblib` depuis la page Models.
- Les **résultats comparatifs** sont exportables en CSV depuis la page Results.
- Les **graphiques** sont exportables en PNG via le bouton dédié sur chaque visualisation.
