"""Exercise the six Streamlit pages using real ML and a local FastAPI test client.

Generated fixtures verify application behavior, not AI4I performance.
"""
from pathlib import Path
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

import api
import core


def widget(elements, label):
    return next(element for element in elements if element.label == label)


def assert_no_errors(app):
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error, [error.value for error in app.error]


def test_six_modules_and_stale_outputs(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    sensor = rng.normal(size=160)
    frame = pd.DataFrame({
        'Type': rng.choice(['H', 'M', 'L'], 160),
        'Air temperature [K]': 300 + sensor,
        'Process temperature [K]': 310 + rng.normal(size=160),
        'Rotational speed [rpm]': 1500 + 100 * rng.normal(size=160),
        'Torque [Nm]': 40 + 5 * rng.normal(size=160),
        'Tool wear [min]': rng.uniform(0, 200, 160),
        core.TARGET: (sensor + rng.normal(scale=.7, size=160) > .7).astype(int),
    })
    monkeypatch.setattr(core, 'load_uci', lambda: frame.copy())
    artifact = tmp_path / 'model.joblib'
    monkeypatch.setattr(core, 'ARTIFACT', artifact)
    monkeypatch.setattr(api, 'ARTIFACT', artifact)
    st.cache_data.clear()
    with TestClient(api.app) as client:
        def http_request(method, url, **kwargs):
            kwargs.pop('timeout', None)
            return client.request(method, urlsplit(url).path, **kwargs)
        monkeypatch.setattr(requests, 'request', http_request)
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=120)
        app.run()
        assert_no_errors(app)
        app.sidebar.radio[0].set_value('2 · Préparation').run()
        assert_no_errors(app)
        app.sidebar.radio[0].set_value('3 · Modèles').run()
        widget(app.multiselect, 'Modèles à entraîner').set_value(['XGBoost'])
        widget(app.button, '🚀 Entraîner et comparer').click().run()
        assert_no_errors(app)
        assert 'XGBoost' in app.session_state['run']['models']
        app.sidebar.radio[0].set_value('4 · Évaluation').run()
        assert_no_errors(app)
        assert len(app.metric) == 7
        app.sidebar.radio[0].set_value('5 · XAI').run()
        widget(app.button, 'Activer ce modèle dans l’API FastAPI').click().run()
        assert_no_errors(app)
        assert artifact.exists()
        assert app.session_state['active_model_id'] == core.load_bundle()['id']
        widget(app.button, 'Calculer SHAP global (échantillon de 25 observations)').click().run()
        assert_no_errors(app)
        assert app.session_state['global_cache'][0] == 'XGBoost'
        widget(app.button, 'Expliquer cet individu (SHAP + LIME)').click().run()
        assert_no_errors(app)
        assert app.session_state['local_cache'][2]['lime']
        widget(app.selectbox, 'Ligne du jeu de test').set_value(1).run()
        assert_no_errors(app)
        assert not app.metric, 'An explanation for the previous observation must disappear'
        app.sidebar.radio[0].set_value('6 · API').run()
        assert_no_errors(app)
        assert any('API connectée' in item.value for item in app.success)
        app.sidebar.radio[0].set_value('Synthèse').run()
        widget(app.button, 'Générer le one-pager PDF avec les résultats calculés').click().run()
        assert_no_errors(app)
        assert app.session_state['onepager_pdf'].startswith(b'%PDF')
        widget(app.slider, 'Seuil retenu pour le one-pager').set_value(.65).run()
        assert_no_errors(app)
        assert 'onepager_pdf' not in app.session_state
