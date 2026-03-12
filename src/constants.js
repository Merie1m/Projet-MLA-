// ─── ML Algorithm Definitions ───────────────────────────────────────────────

export const ML_ALGORITHMS = [
  {
    id: 'svm',
    name: 'Support Vector Machine (SVM)',
    category: 'Classification',
    description:
      'SVM trouve l\'hyperplan optimal qui sépare les classes avec la marge maximale. Efficace en haute dimension et avec des noyaux non-linéaires (RBF, polynomial).',
    hyperparams: [
      { key: 'C', label: 'Régularisation (C)', type: 'number', default: 1.0, min: 0.001, max: 1000, step: 0.1 },
      { key: 'kernel', label: 'Noyau', type: 'select', default: 'rbf', options: ['linear', 'poly', 'rbf', 'sigmoid'] },
      { key: 'gamma', label: 'Gamma', type: 'select', default: 'scale', options: ['scale', 'auto'] },
      { key: 'degree', label: 'Degré (poly)', type: 'number', default: 3, min: 1, max: 10, step: 1 },
    ],
  },
  {
    id: 'random_forest',
    name: 'Random Forest',
    category: 'Ensemble',
    description:
      'Ensemble d\'arbres de décision entraînés sur des sous-échantillons aléatoires. Robuste au sur-apprentissage et gère bien les features catégorielles et numériques.',
    hyperparams: [
      { key: 'n_estimators', label: 'Nombre d\'arbres', type: 'number', default: 100, min: 10, max: 2000, step: 10 },
      { key: 'max_depth', label: 'Profondeur max', type: 'number', default: 10, min: 1, max: 100, step: 1 },
      { key: 'min_samples_split', label: 'Min samples split', type: 'number', default: 2, min: 2, max: 50, step: 1 },
      { key: 'min_samples_leaf', label: 'Min samples leaf', type: 'number', default: 1, min: 1, max: 50, step: 1 },
      { key: 'max_features', label: 'Max features', type: 'select', default: 'sqrt', options: ['sqrt', 'log2', 'auto'] },
    ],
  },
  {
    id: 'knn',
    name: 'K-Nearest Neighbors (KNN)',
    category: 'Classification',
    description:
      'Classifie un point selon la majorité de ses K voisins les plus proches. Simple mais peut être lent sur de grands datasets. Sensible à l\'échelle des features.',
    hyperparams: [
      { key: 'n_neighbors', label: 'K (voisins)', type: 'number', default: 5, min: 1, max: 50, step: 1 },
      { key: 'weights', label: 'Pondération', type: 'select', default: 'uniform', options: ['uniform', 'distance'] },
      { key: 'metric', label: 'Métrique', type: 'select', default: 'minkowski', options: ['minkowski', 'euclidean', 'manhattan', 'chebyshev'] },
      { key: 'p', label: 'Puissance (Minkowski)', type: 'number', default: 2, min: 1, max: 5, step: 1 },
    ],
  },
  {
    id: 'logistic_regression',
    name: 'Régression Logistique',
    category: 'Classification',
    description:
      'Modèle linéaire pour la classification binaire/multi-classe. Interprétable, rapide, et sert de baseline solide. Régularisation L1/L2 disponible.',
    hyperparams: [
      { key: 'C', label: 'Régularisation (C)', type: 'number', default: 1.0, min: 0.001, max: 1000, step: 0.1 },
      { key: 'penalty', label: 'Pénalité', type: 'select', default: 'l2', options: ['l1', 'l2', 'elasticnet', 'none'] },
      { key: 'solver', label: 'Solveur', type: 'select', default: 'lbfgs', options: ['lbfgs', 'liblinear', 'newton-cg', 'saga'] },
      { key: 'max_iter', label: 'Itérations max', type: 'number', default: 100, min: 50, max: 10000, step: 50 },
    ],
  },
  {
    id: 'neural_network',
    name: 'Réseau de Neurones (MLP)',
    category: 'Deep Learning',
    description:
      'Perceptron multi-couches avec rétropropagation. Capable d\'approximer des fonctions complexes non-linéaires. Nécessite plus de données et de calcul.',
    hyperparams: [
      { key: 'hidden_layers', label: 'Couches cachées', type: 'text', default: '128,64,32' },
      { key: 'learning_rate', label: 'Learning rate', type: 'number', default: 0.001, min: 0.00001, max: 1, step: 0.0001 },
      { key: 'activation', label: 'Activation', type: 'select', default: 'relu', options: ['relu', 'tanh', 'sigmoid', 'leaky_relu'] },
      { key: 'optimizer', label: 'Optimiseur', type: 'select', default: 'adam', options: ['adam', 'sgd', 'rmsprop', 'adamw'] },
      { key: 'epochs', label: 'Epochs', type: 'number', default: 50, min: 1, max: 1000, step: 1 },
      { key: 'batch_size', label: 'Batch size', type: 'number', default: 32, min: 1, max: 512, step: 1 },
      { key: 'dropout', label: 'Dropout', type: 'number', default: 0.2, min: 0, max: 0.9, step: 0.05 },
    ],
  },
  {
    id: 'gradient_boosting',
    name: 'Gradient Boosting (XGBoost)',
    category: 'Ensemble',
    description:
      'Boosting séquentiel d\'arbres de décision. Très performant en compétitions ML. Gère les valeurs manquantes nativement.',
    hyperparams: [
      { key: 'n_estimators', label: 'Nombre d\'arbres', type: 'number', default: 200, min: 10, max: 5000, step: 10 },
      { key: 'learning_rate', label: 'Learning rate', type: 'number', default: 0.1, min: 0.001, max: 1, step: 0.01 },
      { key: 'max_depth', label: 'Profondeur max', type: 'number', default: 6, min: 1, max: 50, step: 1 },
      { key: 'subsample', label: 'Subsample', type: 'number', default: 0.8, min: 0.1, max: 1, step: 0.05 },
      { key: 'colsample_bytree', label: 'Colsample bytree', type: 'number', default: 0.8, min: 0.1, max: 1, step: 0.05 },
    ],
  },
  {
    id: 'decision_tree',
    name: 'Arbre de Décision',
    category: 'Classification',
    description:
      'Modèle basé sur des règles if/else apprises depuis les données. Très interprétable, mais peut facilement sur-apprendre sans élagage.',
    hyperparams: [
      { key: 'max_depth', label: 'Profondeur max', type: 'number', default: 10, min: 1, max: 100, step: 1 },
      { key: 'criterion', label: 'Critère', type: 'select', default: 'gini', options: ['gini', 'entropy', 'log_loss'] },
      { key: 'min_samples_split', label: 'Min samples split', type: 'number', default: 2, min: 2, max: 50, step: 1 },
      { key: 'min_samples_leaf', label: 'Min samples leaf', type: 'number', default: 1, min: 1, max: 50, step: 1 },
    ],
  },
  {
    id: 'naive_bayes',
    name: 'Naive Bayes',
    category: 'Classification',
    description:
      'Classificateur probabiliste basé sur le théorème de Bayes. Très rapide, bon pour le NLP et la classification de texte. Suppose l\'indépendance des features.',
    hyperparams: [
      { key: 'var_smoothing', label: 'Lissage variance', type: 'number', default: 1e-9, min: 1e-12, max: 1, step: 1e-9 },
    ],
  },
  // ── Regression algorithms ──────────────────────────────────────────────────
  {
    id: 'linear_regression',
    name: 'Régression Linéaire',
    category: 'Régression',
    description:
      'Modèle linéaire pour la régression. Trouve la droite (ou hyperplan) qui minimise la somme des carrés des résidus. Simple, interprétable et rapide.',
    hyperparams: [],
  },
  {
    id: 'svr',
    name: 'Support Vector Regression (SVR)',
    category: 'Régression',
    description:
      'Version régression du SVM. Utilise des noyaux (RBF, polynomial) pour capturer des relations non-linéaires entre les features et la cible.',
    hyperparams: [
      { key: 'C', label: 'Régularisation (C)', type: 'number', default: 1.0, min: 0.001, max: 1000, step: 0.1 },
      { key: 'kernel', label: 'Noyau', type: 'select', default: 'rbf', options: ['linear', 'poly', 'rbf', 'sigmoid'] },
      { key: 'gamma', label: 'Gamma', type: 'select', default: 'scale', options: ['scale', 'auto'] },
      { key: 'epsilon', label: 'Epsilon', type: 'number', default: 0.1, min: 0.001, max: 1, step: 0.01 },
    ],
  },
  {
    id: 'rf_regression',
    name: 'Random Forest Régression',
    category: 'Régression',
    description:
      'Ensemble d\'arbres de décision pour la régression. Robuste, gère les relations non-linéaires et les interactions entre variables.',
    hyperparams: [
      { key: 'n_estimators', label: 'Nombre d\'arbres', type: 'number', default: 100, min: 10, max: 2000, step: 10 },
      { key: 'max_depth', label: 'Profondeur max', type: 'number', default: 10, min: 1, max: 100, step: 1 },
      { key: 'min_samples_split', label: 'Min samples split', type: 'number', default: 2, min: 2, max: 50, step: 1 },
      { key: 'min_samples_leaf', label: 'Min samples leaf', type: 'number', default: 1, min: 1, max: 50, step: 1 },
    ],
  },
];

// ─── Tuning strategies ──────────────────────────────────────────────────────

export const TUNING_STRATEGIES = [
  { id: 'grid_search', name: 'Grid Search', description: 'Recherche exhaustive de toutes les combinaisons. Lent mais complet.' },
  { id: 'random_search', name: 'Random Search', description: 'Échantillonnage aléatoire de l\'espace. Plus rapide que Grid Search.' },
  { id: 'optuna', name: 'Optuna (Bayesian)', description: 'Optimisation bayésienne intelligente. Le plus efficace pour les grands espaces.' },
];

// ─── Data cleaning operations ───────────────────────────────────────────────

export const CLEANING_OPERATIONS = [
  { id: 'drop_na', label: 'Supprimer les lignes avec valeurs manquantes', icon: 'DeleteSweep' },
  { id: 'fill_mean', label: 'Remplir valeurs manquantes (moyenne)', icon: 'Calculate' },
  { id: 'fill_median', label: 'Remplir valeurs manquantes (médiane)', icon: 'Calculate' },
  { id: 'fill_mode', label: 'Remplir valeurs manquantes (mode)', icon: 'Calculate' },
  { id: 'drop_duplicates', label: 'Supprimer les doublons', icon: 'ContentCopy' },
  { id: 'normalize', label: 'Normaliser (Min-Max)', icon: 'TrendingFlat' },
  { id: 'standardize', label: 'Standardiser (Z-score)', icon: 'TrendingFlat' },
  { id: 'encode_label', label: 'Encoder les labels (Label Encoding)', icon: 'Code' },
  { id: 'encode_onehot', label: 'Encoder en One-Hot', icon: 'ViewModule' },
];

// ─── Metrics ────────────────────────────────────────────────────────────────

export const METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss'];

export const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1_score: 'F1-Score',
  roc_auc: 'ROC AUC',
  log_loss: 'Log Loss',
};

export const REGRESSION_METRICS = ['mae', 'mse', 'rmse', 'r2'];

export const REGRESSION_METRIC_LABELS = {
  mae: 'MAE',
  mse: 'MSE',
  rmse: 'RMSE',
  r2: 'R²',
};

export const METRIC_COLORS = [
  '#2563eb', '#7c3aed', '#16a34a', '#ea580c', '#dc2626', '#0891b2',
  '#ca8a04', '#be185d', '#4f46e5', '#059669',
];

// ─── Dimension reduction ────────────────────────────────────────────────────

export const DIM_REDUCTION_METHODS = [
  { id: 'pca', name: 'PCA', description: 'Analyse en Composantes Principales — projection linéaire rapide.' },
  { id: 'tsne', name: 't-SNE', description: 'T-distributed Stochastic Neighbor Embedding — préserve les voisinages locaux.' },
];

// ─── Regression algorithm IDs ───────────────────────────────────────────────

export const REGRESSION_ALGO_IDS = ['linear_regression', 'svr', 'rf_regression'];
