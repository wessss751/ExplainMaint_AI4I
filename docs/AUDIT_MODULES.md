# Audit du sujet et des livrables — 13 octobre 2026

Référence : [sujet officiel](../assets/sujet_officiel.pdf), pages 2–3, cas n° 15 de l’annexe. L’audit rapproche les exigences du code de l’application et des preuves fournies. Une fonctionnalité présente dans le code ne constitue pas, à elle seule, une validation de son exécution ou de son déploiement.

## Couverture des six modules

| Module demandé | Réalisation dans le projet | Éléments vérifiés |
| --- | --- | --- |
| 1. Données | `app.py` : onglet Données, import CSV, téléchargement et cache via `core.load_uci` | Aperçu, dimensions, statistiques, valeurs manquantes, proportions, histogrammes, corrélations et taux positif par variable. Capture 01 : 10 000 lignes et source UCI. |
| 2. Préparation | `app.py` : choix cible/classe positive/prédicteurs, test de 15 à 40 % ; `core.prepare_binary`, `make_preprocessor`, `train_compare` | Imputation médiane/mode, one-hot, StandardScaler dans un pipeline ajusté sur train après découpage stratifié. Exclusion des identifiants et indicateurs de panne pour AI4I. |
| 3. Modélisation et comparaison | Cinq modèles dans `core.candidate_estimators` ; sélecteurs UI ; `RandomizedSearchCV` optionnel | Régression logistique, arbre, KNN, Random Forest, XGBoost ; paramètres et optimisation 3 plis × 4 tirages. Tableau et export CSV. Capture 02 fournit une exécution historique réelle. |
| 4. Évaluation | `core.threshold_metrics`, onglet Évaluation et notebook | AUC-ROC, AP, précision/rappel/F1 classe 1, ROC superposées, PR, confusion et seuil. Gini/KS concernent le scoring crédit dans le sujet et ne sont pas exigés pour ce cas industriel. |
| 5. XAI | `model_importance`, `permutation_table`, `global_explain`, `local_explain` et onglet XAI | Importances natives, permutation sur test, SHAP beeswarm/dépendance, waterfall local, LIME pour la même observation, saisie fictive ou sélection du test, probabilité, décision et phrase automatique. Capture 05 montre les sorties locales historiques. |
| 6. FastAPI | `api.py` et appels `requests` de `app.py` | Routes `/health`, `/model`, `/global`, `/explain`, OpenAPI `/docs`. Activation du même pipeline sérialisé ; repli local explicitement signalé. Capture 06 montre une API locale connectée avec XGBoost actif. |

La Synthèse ajoute la génération du one-pager ; elle ne remplace pas le sixième module FastAPI.

## Contrôles spécifiques ML / XAI

- **XGBoost** : `XGBClassifier` fait partie du même pipeline que les autres classifieurs ; `scale_pos_weight` est déterminé sur l’entraînement. La recherche d’hyperparamètres reste dans les plis d’entraînement.
- **SHAP** : explication de la probabilité de la classe 1 par permutations, sur variables transformées. Les tests exécutés comparent `base_value + sum(shap_values)` à la probabilité du pipeline, pour les explications locales et globales des cinq modèles.
- **LIME** : même pipeline, même observation et même espace transformé que SHAP ; les coefficients locaux ne sont pas des contributions SHAP. Le désaccord des signes doit être discuté et ne constitue pas automatiquement une erreur.
- **Prévention de fuite** : les modes de panne `TWF/HDF/PWF/OSF/RNF` ainsi que les identifiants `UDI/UID/Product ID` ne doivent pas être des prédicteurs de `Machine failure`. L’interface et le code partagé appliquent cette protection.
- **Limites** : AP est la moyenne des précisions (`average_precision_score`), présentée comme résumé précision-rappel. Le modèle le mieux classé sur le test est un choix descriptif ; une sélection indépendante et une validation temporelle/terrain restent nécessaires. La probabilité n’est pas accompagnée d’une étude de calibration. Les perturbations de variables one-hot peuvent créer des combinaisons absentes des données.

## One-pager et preuves historiques préservés

[`one_pager_final.pdf`](../one_pager_final.pdf) est la synthèse historique à déposer, dont l’unique page a été vérifiée avec `pypdf`. Le document contient le problème, le protocole, les cinq modèles, une explication du même individu par SHAP/LIME, les recommandations, la question des variables sensibles et les limites. Le tableau et l’exemple local concordent avec les captures 02 et 05 inspectées visuellement.

Les chiffres historiques ci-dessous ont été relevés dans ce PDF et dans la capture 02. La nouvelle exécution du notebook les a ensuite reproduits, pour les cinq modèles et les cinq métriques, à la précision affichée de quatre décimales :

| Modèle | AUC-ROC | AP | Précision | Rappel | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0,9766 | 0,7764 | 0,5113 | 0,8000 | 0,6239 |
| Random Forest | 0,9704 | 0,6306 | 0,5385 | 0,7412 | 0,6238 |
| Arbre de décision | 0,9227 | 0,5194 | 0,3020 | 0,9059 | 0,4529 |
| KNN | 0,8554 | 0,5107 | 0,8000 | 0,2824 | 0,4174 |
| Régression logistique | 0,8836 | 0,3833 | 0,1393 | 0,8000 | 0,2373 |

Seuil historique : 0,50. L’exemple local montre environ 2,4 % de probabilité de panne, une référence SHAP d’environ 9,5 %, et des signes SHAP/LIME partiellement divergents. Le PDF décrit ces divergences sans prétendre à une causalité.

`one_pager_modele.pdf` reste un canevas distinct ; il ne remplace pas le PDF final mesuré. Toutes les captures attestent d’une exécution locale (`localhost:8501`), sans preuve de publication. Malgré son nom, `04_shap_global.png` montre l’importance native de la régression logistique ; le beeswarm SHAP n’est pas visible dans cette capture.

Empreinte SHA-256 du PDF final préservé :

```text
2AA24A10109EAF6705A5E9B10869C6142A5F7A77315D6BDDEC145D6A3E52DEEE
```

## Notebook et vérification reproductible

[`notebooks/analyse_ai4i.ipynb`](../notebooks/analyse_ai4i.ipynb) utilise le même code ML que l’application. Corrections apportées : littéral de chaîne invalide dans la cellule locale, tableaux NumPy pour SHAP waterfall, courbes réellement superposées, seuil explicite et comparaison de seuils. Le module API du notebook utilise un artefact temporaire et les vraies routes FastAPI via `TestClient` ; il ne modifie pas le modèle actif de l’application.

L’exécution complète du 13 octobre 2026 a enregistré les 11 cellules avec leurs sorties calculées, sans sortie d’erreur. Les métriques, métadonnées de versions/protocole/empreinte des données, réponse locale, contrôle API et nouveau PDF se trouvent dans `results/validation_2026-10-13/`. Aucun chiffre n’est créé lorsque seule la validation syntaxique est exécutée.

```powershell
.\.venv\Scripts\python.exe scripts/validate_notebook.py
.\.venv\Scripts\python.exe scripts/validate_notebook.py --execute
```

Le validateur contrôle également que les PDF historiques et les fichiers `assets/` restent identiques avant/après exécution.

## Résultats de la nouvelle exécution

- Données effectivement utilisées : 10 000 observations, 339 pannes, 6 prédicteurs ; 7 500 lignes d’entraînement et 2 500 de test, séparation stratifiée, graine 42. Paramètres identiques à ceux des captures, sans optimisation activée.
- [Métriques recalculées](../results/validation_2026-10-13/comparaison_modeles.csv) : les 25 valeurs concordent avec le tableau historique. XGBoost reste premier par AP avec 0,7764, AUC-ROC 0,9766, précision 0,5113, rappel 0,8000 et F1 0,6239 au seuil 0,50.
- [Explication recalculée](../results/validation_2026-10-13/explication_locale.json) : probabilité XGBoost 0,0239780154 et référence SHAP 0,0950079042 pour la première observation du test ; SHAP et LIME calculés sur cette même observation. L’assertion d’additivité SHAP a réussi.
- [API en processus](../results/validation_2026-10-13/api_validation.json) : réponses 200 de `/health`, `/model`, `/global?n=2`, `/docs`, `/openapi.json` et `POST /explain`. Probabilité et décision identiques au calcul direct ; requête à variables manquantes rejetée en 422. L’artefact utilisé est temporaire et isolé du modèle actif de l’application.
- Graphiques calculés et inspectés : [beeswarm SHAP](../results/validation_2026-10-13/shap_global_beeswarm.png) et [dépendance de l’usure](../results/validation_2026-10-13/shap_dependance.png). Ces fichiers sont des exports Matplotlib des sorties du notebook, pas des captures d’une application déployée.
- [Nouveau PDF mesuré](../results/validation_2026-10-13/one_pager_resultats_mesures.pdf) : une page vérifiée, conservée séparément du PDF final historique.

[Versions et protocole exécutés](../results/validation_2026-10-13/execution_metadata.json) : Python 3.11.7, NumPy 2.4.6, pandas 2.3.3, scikit-learn 1.9.1, XGBoost 3.2.0, SHAP 0.51.0, LIME 0.2.0.1 et FastAPI 0.143.0. Ces versions décrivent l’environnement de validation observé, sans garantir que d’autres versions produiront des sorties identiques.

## Statut de validation de cet audit

- Réalisé : lecture du sujet, inspection du code des six modules, extraction du one-pager et contrôle visuel des six captures.
- Réalisé : JSON du notebook et compilation syntaxique de ses 11 cellules de code, sans erreur (`python scripts/validate_notebook.py`).
- Réalisé : notebook exécuté intégralement, compteurs 1 à 11, aucune sortie d’erreur, format `nbformat` valide. Les deux graphiques SHAP supplémentaires et les résultats viennent de cette exécution.
- Réalisé : 48 tests ML/XAI/FastAPI, sans échec, erreur ni test ignoré, dans le [rapport JUnit](../results/validation_2026-10-13/pytest_ml_api.xml). Les données synthétiques des tests servent à vérifier le code ; seules les métriques AI4I du notebook sont des résultats de comparaison du projet.
- Réalisé : PDF final historique et nouveau PDF mesuré d’une page chacun ; empreinte du PDF final inchangée.
- En attente : vérification d’une URL publique après publication autorisée. Le succès du client de test local ne prouve pas un déploiement public.

Le statut de publication et les éventuelles vérifications ultérieures figurent dans la [checklist de dépôt](../CHECKLIST_DEPOT.md). L’existence de fichiers Docker ou de tests préparés ne prouve pas leur exécution réussie.
