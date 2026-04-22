# Guide de Test — Random Forest, MLflow, et Comparaison Algorithmes

## État des Serveurs
✅ **Frontend :** http://localhost:3000  
✅ **Backend :** http://127.0.0.1:8000  
📂 **Dataset de test :** `test_iris.csv` (dans le dossier racine du projet)

---

## 🎯 Parcours de Test — 5 Étapes

### Étape 1️⃣ : Upload du Dataset
1. Ouvre http://localhost:3000 dans le navigateur
2. Clique sur l'onglet **"Données"** (sidebar gauche)
3. Zone **"Charger un Dataset"** → Drag & drop `test_iris.csv` ou clique pour sélectionner
4. Tu verras :
   - Aperçu des 5 colonnes : `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, **`species`** (cible)
   - Stats : total 45 lignes, 3 classes (setosa, versicolor, virginica)
   - Types détectés : 4 numériques + 1 string (cible)

**🔍 À vérifier :** Les colonnes et le nombre de lignes s'affichent correctement

---

### Étape 2️⃣ : Sélectionner Random Forest
1. Clique sur l'onglet **"Modèles"** (sidebar)
2. Tu verras une liste d'algorithmes avec des cartes colorées
3. Cherche **"Random Forest"** (catégorie : Ensemble)
4. Clique sur la carte pour **cocher/sélectionner**
5. *Optionnel* : Clique sur l'icône ⚙️ pour ajuster les hyperparamètres :
   - `n_estimators` (nombre d'arbres) : défaut 100
   - `max_depth` (profondeur max) : défaut 10
   - `min_samples_split` : défaut 2

**🔍 À vérifier :** Random Forest est bien listé et sélectionnable

---

### Étape 3️⃣ : Lancer l'Entraînement
1. Clique sur l'onglet **"Entraînement"** (sidebar)
2. Tu dois voir un résumé :
   - **Dataset :** test_iris.csv
   - **Modèles** : 1 (Random Forest)
   - **Statut** : Prêt
3. Clique sur le bouton bleu **"🚀 Lancer l'entraînement"**
4. Tu verras une barre de progression avec les étapes :
   - ✓ Validation des données
   - ✓ Pré-traitement
   - ✓ Entraînement
   - ✓ Évaluation
   - ✓ Terminé

**🔍 À vérifier :**
- La barre progresse jusqu'à 100%
- Un toast (notification) apparaît en bas à droite : *"Random Forest — Entraînement terminé (accuracy: 96.4%)"* ou similaire
- Pas d'erreur dans la console du navigateur (F12)

---

### Étape 4️⃣ : Afficher les Résultats
1. Clique sur l'onglet **"Résultats"** (sidebar)
2. Tu verras plusieurs onglets en haut :
   - **Comparaison** (bar chart)
   - **Matrice de confusion**
   - **Courbes ROC**
   - **Courbes PR**
   - **Radar**
   - **Tableau**

**🔍 À vérifier :**
- Un graphique en barres montre la **Accuracy** de Random Forest (~96%)
- La **matrice de confusion** affiche une grille 3×3 avec peu d'erreurs (diagonale verte, off-diag rouge)
- Les **courbes ROC** et **PR** s'affichent (pour classification binaire vs multiclass)
- Un **tableau récapitulatif** en bas avec colonnes : Model | Accuracy | Precision | Recall | F1 | ROC AUC | ...

---

### Étape 5️⃣ : Vérifier MLflow
1. Clique sur l'onglet **"MLOps"** (sidebar)
2. Tu dois voir une **table d'expériences** avec une ligne pour Random Forest contenant :
   - **Algorithme** : Random Forest
   - **Dataset** : test_iris.csv
   - **Métriques** : Accuracy, Precision, Recall, F1, ROC AUC
   - **Type** : classification
   - **MLflow Run** : un petit chip avec l'ID tronqué (ex. `a1b2c3d4`)
3. *Optionnel* : Clique sur la ligne pour voir plus de détails dans une modal

**🔍 À vérifier :**
- L'expérience est enregistrée avec MLflow
- Les paramètres et métriques sont affichés
- L'ID du run MLflow est présent

---

## 🔀 Bonus : Comparer avec l'Arbre de Décision

1. Retourne sur **"Modèles"**
2. Cherche et sélectionne également **"Arbre de Décision"** (Decision Tree)
3. Retourne à **"Entraînement"**
4. Tu verras maintenant 2 modèles sélectionnés
5. Lance l'entraînement
6. Va dans **"Résultats"** pour voir **les deux algorithmes côte à côte** :
   - Bar chart comparant Accuracy, Precision, Recall, F1
   - Matrice de confusion pour chacun
   - Courbes ROC/PR pour les deux

**🔍 À vérifier :**
- Random Forest a généralement une **accuracy supérieure** à Decision Tree
- La comparaison visuelle est claire
- Les deux runs apparaissent dans MLOps

---

## 📊 Tableau Récapitulatif des Points Testés

| Fonctionnalité | Implémentée | Test | Où le Voir |
|---|---|---|---|
| **Random Forest Classification** | ✅ | Entraînement s'exécute | Résultats / Accuracy |
| **Random Forest Regression** | ✅ | À tester avec données continues | Même processus, métriques MAE/MSE/R² |
| **Sélection d'algorithmes** | ✅ | Drag-select dans Modèles | Page Modèles |
| **Hyperparamètres configurables** | ✅ | Ajuster n_estimators, max_depth | Page Modèles, cartes d'algo |
| **MLflow Tracking** | ✅ | Run ID et métriques loggées | Page MLOps |
| **Matrice de confusion** | ✅ | Affichée après entraînement | Résultats → Onglet "Matrice" |
| **Courbes ROC / PR** | ✅ | Affichées pour classification | Résultats → Onglets "ROC" / "PR" |
| **Comparaison RF vs DT** | ✅ | Entraîner les deux, comparer | Résultats → Bar chart côte à côte |
| **MLOps / Historique** | ✅ | Table avec tous les runs | Page MLOps |

---

## ❌ Points Pas Encore Implémentés (à faire pour Tâche 4)

| Fonctionnalité | Status |
|---|---|
| **Importance des variables** (feature_importances_) | ❌ À ajouter |
| **Analyse de stabilité** (random_state différents) | ❌ À ajouter |
| **Analyse d'erreurs** (exemples mal classés) | ❌ À ajouter |
| **Tableau biais/variance** (n_estimators vs accuracy) | ❌ À ajouter |
| **Test normalité des résidus** (régression) | ❌ À ajouter |
| **Hétéroscédasticité** (régression) | ❌ À ajouter |

---

## 🐛 Troubleshooting

### Frontend 404 ou ne charge pas
- Vérifie que `npm run dev` est bien lancé dans le terminal
- L'URL doit être http://localhost:3000 (pas 5173 ni 8000)

### Backend Erreur 500
- Vérifie que le backend démarre sans erreur (voir terminal avec `uvicorn`)
- Check `backend/requirements.txt` : tous les packages sont-ils installés ? Vire run `pip install -r backend/requirements.txt`

### Entraînement échoue
- Vérifie que le dataset est bien uploadé et a au moins une colonne cible
- Regarde la console navigateur (F12) pour voir les messages d'erreur

### MLflow Run ID vide
- Normal si l'entraînement a échoué ; réessaye après correction

---

## ✨ Conclusion

Les points **1–5 ci-dessus** te permettront de voir que :
- ✅ Random Forest est pleinement implémenté en classification
- ✅ MLflow enregistre et récupère les runs
- ✅ Le frontend affiche les résultats, comparaisons et courbes
- ✅ La comparaison avec Decision Tree fonctionne

Les points **❌** sont à développer pour compléter la Tâche 4.

Bon test ! 🎉
