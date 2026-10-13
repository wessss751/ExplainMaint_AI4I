# Déployer ExplainMaint sur Hugging Face Spaces

Le Dockerfile expose **Streamlit et FastAPI sur le même port public 7860**. Nginx
transmet `/` et les WebSockets à Streamlit (`127.0.0.1:8501`), et `/api/*` à
FastAPI (`127.0.0.1:8000`). Cette organisation suit la recommandation des
[Docker Spaces pour plusieurs services](https://huggingface.co/docs/hub/spaces-sdks-docker).

| Ressource | Chemin public |
| --- | --- |
| Interface Streamlit | `/` |
| Santé Streamlit | `/_stcore/health` |
| Santé FastAPI | `/api/health` |
| Modèle actif | `/api/model` |
| Explication locale SHAP + LIME | `POST /api/explain` |
| Explication globale SHAP | `/api/global?n=25` |
| Documentation Swagger | `/api/docs` |
| Schéma OpenAPI | `/api/openapi.json` |

`EXPLAINMAINT_API_ROOT_PATH=/api` est défini dans le Dockerfile : Swagger utilise
ainsi les URL publiques correctes. L'application appelle toujours le service
interne via `EXPLAINMAINT_API_URL=http://127.0.0.1:8000`. Un démarrage local sans
Docker conserve `/docs` sur le port 8000. Voir la documentation
[FastAPI derrière un proxy](https://fastapi.tiangolo.com/advanced/behind-a-proxy/).

Sur Spaces, `SPACE_HOST` configure automatiquement l'adresse publique Streamlit
(port HTTPS 443), ce qui conserve ses protections CORS/XSRF. Pour un domaine
personnalisé, définir `STREAMLIT_BROWSER_SERVER_ADDRESS` et, si nécessaire,
`STREAMLIT_BROWSER_SERVER_PORT`. Voir les
[variables automatiques Spaces](https://huggingface.co/docs/hub/spaces-overview#built-in-environment-variables)
et la [configuration d'adresse Streamlit](https://docs.streamlit.io/knowledge-base/deploy/remote-start).

## Vérification Docker locale

Démarrer Docker Desktop en mode conteneurs Linux, puis, depuis le projet :

```powershell
docker build -t explainmaint-ai4i .
docker run --rm --name explainmaint-ai4i -p 7860:7860 explainmaint-ai4i
```

Dans un autre terminal, avec l'environnement Python du projet activé :

```powershell
python scripts/verify_deployment.py http://localhost:7860 --output results/docker_services_check.json
```

Le contrôle vérifie le HTML, les deux services, la connexion WebSocket, Swagger
et les métadonnées du modèle. Sans modèle actif, le rapport indique explicitement
`scope: services_only` et ne valide pas encore les explications.

Dans Streamlit, charger AI4I, préparer les données, entraîner les modèles, puis
activer le modèle choisi dans « XAI ». Relancer ensuite :

```powershell
python scripts/verify_deployment.py http://localhost:7860 --require-model --output results/docker_full_check.json
```

Ce second contrôle appelle réellement `/explain` et `/global`, vérifie SHAP
(dont son additivité), LIME, la probabilité et la décision au seuil 0,5. Il utilise
un enregistrement de démonstration avec les six variables AI4I habituelles.
Pour un modèle avec d'autres variables, fournir `--record-json chemin/record.json`.
Ce contrôle technique ne constitue pas une mesure de performance du modèle.

Le conteneur s'exécute avec l'utilisateur 1000. Tini et `start.sh` arrêtent tous
les services si l'un d'eux tombe ; un `HEALTHCHECK` traverse Nginx et contrôle les
deux services. Les en-têtes WebSocket sont transmis explicitement, conformément
à la [documentation Nginx](https://nginx.org/en/docs/http/websocket.html).

## Préparer les fichiers à publier

Le README principal contient les métadonnées `sdk: docker` et `app_port: 7860`.
Pour produire un dossier d'envoi et son manifeste SHA-256 :

```powershell
python scripts/prepare_space.py
```

Le dossier `dist/huggingface-space` contient le code, le notebook, les captures,
la configuration et les PDF. `one_pager_final.pdf` est copié sans modification.
Le script sélectionne explicitement les livrables ; il exclut les environnements,
les caches, les secrets et les modèles locaux. Il refuse d'écraser un dossier
non vide. Pour refaire un paquet, choisir un nouveau chemin avec `--output`.
Cette commande locale n'authentifie aucun compte et ne publie rien.

## Publication après autorisation

La publication nécessite un compte Hugging Face et son autorisation. Le jeton
n'est pas nécessaire au fonctionnement de l'application et n'a pas à figurer
dans le dépôt ni dans le chat. Avec le CLI Hugging Face installé :

```powershell
python -m pip install huggingface_hub
hf auth login
hf auth whoami
```

Choisir le compte et le nom du Space, puis remplacer `VOTRE_COMPTE` ci-dessous.
Les deux commandes suivantes créent et publient un **Space public** :

```powershell
hf repos create VOTRE_COMPTE/ExplainMaint_AI4I --repo-type space --space-sdk docker --public
hf upload VOTRE_COMPTE/ExplainMaint_AI4I dist/huggingface-space . --repo-type space
```

Pour un Space existant, passer directement à l'envoi après avoir vérifié son
propriétaire et son nom. Ces commandes suivent la
[référence officielle du CLI](https://huggingface.co/docs/huggingface_hub/package_reference/cli).
L'interface « New Space » permet également de choisir **Docker**, puis d'envoyer
les fichiers préparés dans l'onglet « Files ».

Attendre le statut « Running » dans le Space. Copier son adresse réelle
`https://…hf.space` depuis l'interface, puis exécuter les mêmes contrôles avec
cette adresse à la place de `http://localhost:7860`. Après activation du modèle,
le contrôle avec `--require-model` doit réussir. Conserver le JSON obtenu et
ouvrir `/api/docs` pour vérifier la documentation interactive dans le navigateur.
Le lien public doit être reporté dans le README uniquement après cette vérification.

## Données et résultats

Aucun modèle pré-entraîné n'est livré. Les données UCI se chargent depuis
l'application et le modèle actif est créé dans `artifacts/active_bundle.joblib`.
Le stockage du conteneur est éphémère : un redémarrage peut nécessiter un nouvel
entraînement et une nouvelle activation. Les mesures originales du PDF restent
les résultats de leur exécution d'origine ; une nouvelle exécution ne les remplace
pas. Le service est un démonstrateur universitaire.
