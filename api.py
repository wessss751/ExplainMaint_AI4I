"""Interpretability REST service: OpenAPI at http://localhost:8000/docs."""
import os
from typing import Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from core import ARTIFACT, global_explain, load_bundle, local_explain

app = FastAPI(title='ExplainMaint XAI API', version='1.0.0',
              root_path=os.environ.get('EXPLAINMAINT_API_ROOT_PATH', ''),
              description='Prédiction et explications SHAP / LIME du modèle sélectionné dans Streamlit.')

class ExplainRequest(BaseModel):
    record: dict[str, Any] = Field(..., description='Valeurs de toutes les variables explicatives du modèle actif')
    threshold: float = Field(default=.5, ge=0.0, le=1.0)
    expected_model_id: str | None = Field(default=None, description='Identifiant renvoyé par /model ; détecte une activation concurrente')


def check_model_id(bundle, expected_model_id):
    if expected_model_id is not None and expected_model_id != bundle.get('id'):
        raise HTTPException(409, 'Un autre modèle a été activé. Réactiver le modèle souhaité ou calculer localement.')

@app.get('/health')
def health():
    return {'status': 'ok', 'trained_model_available': ARTIFACT.exists()}

@app.get('/model')
def model_info():
    try:
        bundle = load_bundle()
        return {'name': bundle['name'], 'features': bundle['features'],
                'reference_count': len(bundle['background']), 'model_id': bundle.get('id')}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@app.post('/explain')
def explain(request: ExplainRequest):
    try:
        bundle = load_bundle()
        check_model_id(bundle, request.expected_model_id)
        result = local_explain(bundle, request.record)
        result['model_id'] = bundle.get('id')
        result['threshold'] = request.threshold
        result['decision'] = int(result['probability'] >= request.threshold)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc

@app.get('/global')
def explain_global(n: int = Query(default=25, ge=1, le=60),
                   expected_model_id: str | None = Query(default=None)):
    try:
        bundle = load_bundle()
        check_model_id(bundle, expected_model_id)
        result = global_explain(bundle, n=n)
        result['model_id'] = bundle.get('id')
        return result
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
