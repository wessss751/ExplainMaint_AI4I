# Checklist finale — 13 octobre 2026

## Livrables universitaires

- [x] Six modules rapprochés du sujet officiel : [audit détaillé](docs/AUDIT_MODULES.md).
- [x] Code complet Streamlit, FastAPI, ML/XAI et génération PDF.
- [x] Cinq modèles testés, dont XGBoost et optimisation par validation croisée.
- [x] SHAP global/local et LIME exécutés ; additivité SHAP contrôlée.
- [x] FastAPI testée : routes, OpenAPI, proxy, validation et conflits d’activation.
- [x] Parcours des six pages Streamlit testé, avec entraînement et génération PDF.
- [x] **49 tests réussis** : [rapport JUnit](results/validation_2026-10-13/pytest_final.xml).
- [x] Notebook exécuté intégralement, **11 cellules sans erreur**, sorties conservées.
- [x] Les 25 métriques recalculées reproduisent les résultats fournis à quatre décimales : [CSV](results/validation_2026-10-13/comparaison_modeles.csv).
- [x] `one_pager_final.pdf` conservé : **c’est le PDF à remettre**, une page.
- [x] Six captures originales et sujet officiel préservés à l’octet près : [empreintes](docs/original_artifacts.sha256.json).
- [x] Nouvelles figures SHAP calculées enregistrées séparément dans `results/validation_2026-10-13/`.
- [x] README, notebook, dépendances de production/développement et versions verrouillées fournis.
- [x] Dépôt Git local initialisé sur `main`, exclusions des caches, secrets et modèles générés.
- [x] Configuration CI GitHub préparée ; son exécution distante attend la publication.

## Déploiement

- [x] Docker/Nginx préparés : Streamlit `/`, FastAPI `/api/*`, Swagger `/api/docs`, WebSocket.
- [x] Commandes de préparation du paquet et de vérification HTTP fournies.
- [ ] Construction et vérification locale Docker : en cours de validation.
- [ ] Publication GitHub : accès au compte `wessss751` vérifié, autorisation de publication encore requise.
- [ ] Publication Hugging Face : connexion et autorisation requises ; nom fourni `wissem` à confirmer par connexion.
- [ ] Vérification de l’application publique : nécessite une publication effective.
- [ ] Liens publics vérifiés dans le README : aucun lien de projet publié à ce stade.

## Refaire les contrôles

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_delivery.py
.\.venv\Scripts\python.exe scripts/validate_notebook.py --execute
docker build -t explainmaint-ai4i .
docker run --rm -p 7860:7860 explainmaint-ai4i
# Dans un second terminal, après activation du modèle via l’interface :
.\.venv\Scripts\python.exe scripts/verify_deployment.py http://localhost:7860 --require-model
```

## Lecture des preuves

Les PDF et captures d’origine prouvent une exécution locale historique. Les fichiers de `results/validation_2026-10-13/` proviennent des validations supplémentaires de ce jour. `one_pager_resultats_mesures.pdf` est distinct du PDF final initial. Aucun document local ne prouve à lui seul une publication sur Internet.

Les premiers tests lancés depuis un sous-agent ont rencontré une restriction Windows sur les processus. La suite finale a été relancée depuis la session principale avec la configuration définitive (`n_jobs=-1`) et passe. Les avertissements résiduels sont des dépréciations SHAP/Matplotlib et pandas ; ils ne sont pas des erreurs d’exécution.

## Limites à présenter au jury

Jeu UCI synthétique ; split aléatoire ; choix descriptif sur les scores de test ; aucune validation industrielle ni calibration des probabilités. Les indicateurs de modes de panne sont exclus pour éviter la fuite de cible. SHAP/LIME expliquent le modèle sans établir une causalité. Le seuil relève d’un arbitrage entre pannes manquées et inspections inutiles. Un Space partagé conserve un seul modèle actif, avec détection des conflits et repli local signalé.
