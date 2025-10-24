# Classification d'Images avec un Réseau de Neurones Convolutif (CNN)

## Contexte du Projet

Ce projet implémente un **réseau de neurones convolutif (CNN)** pour la classification d'images du dataset **Fashion-MNIST** à l'aide de **TensorFlow** et **Keras**.  
L'objectif est de reconnaître automatiquement des articles vestimentaires à partir d'images en niveaux de gris.

---

## Stack Technique

- **Langage :** Python  
- **Bibliothèques ML :** TensorFlow / Keras  
- **Analyse et visualisation :** NumPy, Matplotlib, Seaborn  
- **Outils :** Google Colab, Jira (planification et suivi du projet)

---

## Contenu du Projet

### 1. Chargement et Préparation des Données

- Chargement du dataset **Fashion-MNIST** via `tf.keras.datasets.fashion_mnist`
- Visualisation d'exemples d'images et de leurs labels
- Normalisation des pixels entre `0` et `1`
- Séparation des ensembles :
  - **Entraînement :** 55 000 images  
  - **Validation :** 5 000 images  
  - **Test :** 10 000 images  
- Redimensionnement pour convenir à la couche `Conv2D` : `(28, 28, 1)`

---

### 2. Dictionnaire des Classes

| Label | Nom personnalisé |
|:------:|------------------|
| 0 | Tatishirt |
| 1 | Taserwalt |
| 2 | Pullover |
| 3 | Talkswat |
| 4 | Tal9biyat |
| 5 | Tatalont |
| 6 | Tatricot |
| 7 | Tasberdilat |
| 8 | Tasakt |
| 9 | Boot |

---

### 3. Structure du Répertoire

```bash
├── notebook.ipynb    # Notebook Google Colab contenant le code complet
├── README.md          # Documentation du projet
```
---
### 4. Architecture du modèle CNN

Le modèle est construit avec l'API **Sequential** de Keras :

| Couche | Type | Détails |
|:-------|:------|:--------|
| 1 | `Conv2D` | 64 filtres, taille 2x2, activation ReLU |
| 2 | `MaxPooling2D` | Réduction de taille 2x2 |
| 3 | `Dropout` | 30% des neurones désactivés |
| 4 | `Conv2D` | 32 filtres, taille 2x2, activation ReLU |
| 5 | `MaxPooling2D` | Réduction de taille 2x2 |
| 6 | `Dropout` | 30% |
| 7 | `Flatten` | Conversion des features en vecteur |
| 8 | `Dense` | 256 neurones, activation ReLU |
| 9 | `Dropout` | 50% |
| 10 | `Dense` (sortie) | 10 neurones, activation Softmax |

---

## Préparation des Données

Le jeu de données utilisé est **Fashion-MNIST** (60 000 images d'entraînement et 10 000 images de test de 28x28 pixels).

### 5. Processus Clé

1.  **Chargement :** Via `tensorflow.keras.datasets.fashion_mnist`.
2.  **Normalisation :** Les valeurs des pixels ont été divisées par $255.0$ pour obtenir des valeurs en virgule flottante dans l'intervalle $[0, 1]$, améliorant la convergence du modèle.
3.  **Labels :** Les labels sont conservés sous forme **d'entiers** (0 à 9), ce qui dicte l'utilisation de la fonction de perte `sparse_categorical_crossentropy`.

---

### 6. Compilation et entraînement

- **Fonction de perte :** `sparse_categorical_crossentropy`  
- **Optimiseur :** `Adam`  
- **Métrique :** `accuracy`

Deux *callbacks* sont utilisés :
- `ModelCheckpoint` : sauvegarde du meilleur modèle.
- `EarlyStopping` : arrêt automatique en cas de stagnation de la performance sur la validation.

---

### 7. Détails de l'Architecture (résumé technique)

| Couche | Type | Shape de Sortie | Activation | Rôle |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `Flatten` | (None, 784) | N/A | Convertit $28 \times 28$ en vecteur $784$. |
| 2 | `Dense` | (None, **128**) | **ReLU** | Couche cachée pour l'apprentissage des features. |
| 3 | `Dense` | (None, **10**) | **Softmax** | Couche de sortie (10 classes), renvoie les probabilités. |



### ⚡ Bonus : EarlyStopping

Un callback `EarlyStopping` a été mis en place sur la métrique `val_loss` avec une patience de $3$ époques. Ceci a permis d'arrêter l'entraînement automatiquement lorsque la généralisation du modèle cessait de s'améliorer, prévenant ainsi le **surapprentissage (overfitting)**.

---

## Résultats et Analyse

### 8. Précision Finale

Le modèle a atteint une précision de **91%** sur le jeu de test après **10** époques (grâce à l'EarlyStopping). L'objectif MVP est atteint.

---


### 9. Évaluation du Modèle

Après entraînement, le modèle est évalué sur l’ensemble de test.

```python
Test loss: 0.23  
Test accuracy: 0.91
```