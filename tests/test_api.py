"""Exercise the REST contract with a real, temporary XGBoost/SHAP/LIME bundle."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api
import core


@pytest.fixture(scope='module')
def api_bundle():
    rng = np.random.default_rng(42)
    sensor = rng.normal(size=100)
    frame = pd.DataFrame({'sensor': sensor, 'type': rng.choice(['H', 'L'], 100),
                          'outcome': (sensor + rng.normal(scale=.5, size=100) > .4).astype(int)})
    X, y = core.prepare_binary(frame, 'outcome', ['sensor', 'type'], '1')
    run = core.train_compare(X, y, ['XGBoost'], params={'trees': 12, 'depth': 2})
    return {'pipeline': run['models']['XGBoost'], 'background': run['X_train'],
            'features': ['sensor', 'type'], 'name': 'XGBoost'}


@pytest.fixture
def client(api_bundle, monkeypatch, tmp_path):
    artifact = tmp_path / 'active_bundle.joblib'
    monkeypatch.setattr(core, 'ARTIFACT', artifact)
    monkeypatch.setattr(api, 'ARTIFACT', artifact)
    core.store_bundle(api_bundle['pipeline'], api_bundle['background'], api_bundle['features'], api_bundle['name'])
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_model_and_openapi(client):
    assert client.get('/health').json() == {'status': 'ok', 'trained_model_available': True}
    model = client.get('/model')
    assert model.status_code == 200
    assert model.json()['name'] == 'XGBoost'
    assert model.json()['features'] == ['sensor', 'type']
    assert client.get('/docs').status_code == 200
    schema = client.get('/openapi.json').json()
    assert {'/health', '/model', '/global', '/explain'}.issubset(schema['paths'])
    n_schema = schema['paths']['/global']['get']['parameters'][0]['schema']
    assert n_schema['minimum'] == 1 and n_schema['maximum'] == 60


def test_explain_returns_real_shap_lime_and_applies_threshold(client, api_bundle):
    record = {'sensor': 0.3, 'type': 'H'}
    expected = api_bundle['pipeline'].predict_proba(pd.DataFrame([record]))[0, 1]
    response = client.post('/explain', json={'record': record, 'threshold': .7})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['probability'] == pytest.approx(expected)
    assert result['base_value'] + sum(result['shap_values']) == pytest.approx(expected, abs=1e-6)
    assert result['decision'] == int(expected >= .7)
    assert result['threshold'] == .7
    assert result['lime']
    assert all(np.isfinite(item['weight']) for item in result['lime'])


def test_global_explanations(client):
    model_id = client.get('/model').json()['model_id']
    response = client.get('/global', params={'n': 2, 'expected_model_id': model_id})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result['values']) == len(result['base_values']) == len(result['data']) == 2
    assert len(result['names']) == len(result['values'][0])
    assert result['model_id'] == model_id


def test_concurrent_activation_returns_409_before_explaining(client, monkeypatch):
    def unexpected_explanation(*args, **kwargs):
        raise AssertionError('An explanation must not be computed for a different model')

    monkeypatch.setattr(api, 'local_explain', unexpected_explanation)
    monkeypatch.setattr(api, 'global_explain', unexpected_explanation)
    response = client.post('/explain', json={
        'record': {'sensor': .3, 'type': 'H'}, 'expected_model_id': 'previous-activation'})
    assert response.status_code == 409
    assert client.get('/global', params={'n': 2, 'expected_model_id': 'previous-activation'}).status_code == 409


@pytest.mark.parametrize('n', ['0', '-1', '61', '1.5', 'invalid'])
def test_global_rejects_invalid_sample_sizes(client, n):
    assert client.get(f'/global?n={n}').status_code == 422


@pytest.mark.parametrize('payload', [
    {},
    {'record': {'sensor': .3}},
    {'record': {'sensor': .3, 'type': 'H', 'extra': 1}},
    {'record': {'sensor': 'invalid', 'type': 'H'}},
    {'record': {'sensor': 'Infinity', 'type': 'H'}},
    {'record': {'sensor': [1, 2], 'type': 'H'}},
    {'record': {'sensor': .3, 'type': {'category': 'H'}}},
    {'record': {'sensor': .3, 'type': 'H'}, 'threshold': -0.1},
    {'record': {'sensor': .3, 'type': 'H'}, 'threshold': 1.1},
])
def test_explain_validation_returns_422(client, payload):
    assert client.post('/explain', json=payload).status_code == 422


def test_no_active_model_returns_404(client, monkeypatch, tmp_path):
    missing = tmp_path / 'not_trained.joblib'
    monkeypatch.setattr(core, 'ARTIFACT', missing)
    monkeypatch.setattr(api, 'ARTIFACT', missing)
    assert client.get('/health').json()['trained_model_available'] is False
    assert client.get('/model').status_code == 404
    assert client.get('/global').status_code == 404
    assert client.post('/explain', json={'record': {'sensor': .3, 'type': 'H'}}).status_code == 404


def test_docs_honor_reverse_proxy_prefix():
    # ASGI root_path is what FastAPI receives behind the Docker /api proxy.
    with TestClient(api.app, root_path='/api') as proxy_client:
        response = proxy_client.get('/docs')
        assert response.status_code == 200
        assert '/api/openapi.json' in response.text
