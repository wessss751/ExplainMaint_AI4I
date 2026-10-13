"""Unit tests on freshly generated synthetic records; not AI4I performance evidence."""
import numpy as np
import pandas as pd
import pytest
import core
from core import (prepare_binary, train_compare, threshold_metrics, model_importance,
                  store_bundle, load_bundle, local_explain, global_explain)
from report import create_onepager


def sample_df(n=160):
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, n)
    y = (x + rng.normal(0, .7, n) > 0.7).astype(int)
    return pd.DataFrame({'sensor': x, 'type': rng.choice(['H','M','L'], n), 'outcome': y})


@pytest.fixture(scope='module')
def trained_run():
    df = sample_df()
    X, y = prepare_binary(df, 'outcome', ['sensor', 'type'], '1')
    names = ['Régression logistique', 'Arbre de décision', 'KNN', 'Random Forest', 'XGBoost']
    return train_compare(X, y, names, .25, params={'trees': 40, 'depth':3, 'neighbors':5})


def test_binary_and_all_five_models(trained_run):
    run = trained_run
    names = ['Régression logistique', 'Arbre de décision', 'KNN', 'Random Forest', 'XGBoost']
    assert list(run['models']) == names
    assert run['X_train'].index.intersection(run['X_test'].index).empty
    assert all(0 <= run['scores'][name]['AUC-ROC'] <= 1 for name in names)
    assert all(len(run['probs'][name]) == len(run['X_test']) for name in names)
    assert len(model_importance({'pipeline': run['models']['Random Forest'], 'background': run['X_train']})) > 0
    a = threshold_metrics(run['y_test'], run['probs'][names[0]], .2)
    b = threshold_metrics(run['y_test'], run['probs'][names[0]], .8)
    assert a['Rappel'] >= b['Rappel']


def test_preprocessing_fits_only_training_rows(trained_run):
    run = trained_run
    for pipe in run['models'].values():
        imputer = pipe.named_steps['preprocess'].named_transformers_['num'].named_steps['impute']
        assert imputer.statistics_[0] == pytest.approx(run['X_train']['sensor'].median())
        scale = pipe.named_steps['preprocess'].named_transformers_['num'].named_steps['scale']
        assert scale.mean_[0] == pytest.approx(run['X_train']['sensor'].mean())


def test_xgboost_cross_validation_search_runs_on_training_only():
    df = sample_df()
    X, y = prepare_binary(df, 'outcome', ['sensor', 'type'], '1')
    run = train_compare(X, y, ['XGBoost'], params={'trees': 12}, tune=True)
    assert run['X_train'].index.intersection(run['X_test'].index).empty
    assert set(run['tuned']['XGBoost']) == {'model__max_depth', 'model__learning_rate', 'model__subsample'}
    assert len(run['probs']['XGBoost']) == len(run['y_test'])


def test_prepare_rejects_multiclass_and_missing_positive():
    df = sample_df()
    with pytest.raises(ValueError, match='positive'):
        prepare_binary(df, 'outcome', ['sensor', 'type'], 'missing')
    df.loc[:5, 'outcome'] = 2
    with pytest.raises(ValueError, match='exactement deux'):
        prepare_binary(df, 'outcome', ['sensor', 'type'], '1')


@pytest.mark.parametrize('feature', core.LEAKAGE + core.IDENTIFIERS)
def test_ai4i_leakage_is_blocked_for_csv_too(feature):
    df = sample_df().rename(columns={'outcome': core.TARGET})
    df[feature] = 1
    with pytest.raises(ValueError, match='fuite de cible'):
        prepare_binary(df, core.TARGET, ['sensor', feature], '1')


def make_bundle(run, name='XGBoost'):
    return {'pipeline': run['models'][name], 'background': run['X_train'].head(80),
            'features': ['sensor', 'type'], 'name': name}


@pytest.mark.parametrize('name', ['Régression logistique', 'Arbre de décision', 'KNN', 'Random Forest', 'XGBoost'])
def test_real_shap_lime_explanations_all_models(trained_run, name):
    bundle = make_bundle(trained_run, name)
    row = trained_run['X_test'].iloc[0]
    local = local_explain(bundle, row.to_dict())
    expected = bundle['pipeline'].predict_proba(row.to_frame().T)[0, 1]
    assert local['probability'] == pytest.approx(expected)
    assert local['base_value'] + sum(local['shap_values']) == pytest.approx(expected, abs=1e-6)
    assert len(local['shap_values']) == len(local['encoded_values']) == len(local['names'])
    assert len(local['lime']) == len(local['names'])
    assert all(np.isfinite(item['weight']) for item in local['lime'])
    global_result = global_explain(bundle, n=2)
    reconstructed = np.asarray(global_result['base_values']) + np.asarray(global_result['values']).sum(axis=1)
    np.testing.assert_allclose(reconstructed, bundle['pipeline'].predict_proba(bundle['background'].head(2))[:, 1], atol=1e-6)


def test_local_explanation_matches_missing_value_imputation(trained_run):
    bundle = make_bundle(trained_run)
    record = {'sensor': None, 'type': None}
    local = local_explain(bundle, record)
    row = pd.DataFrame({'sensor': [np.nan], 'type': [np.nan]})
    assert local['probability'] == pytest.approx(bundle['pipeline'].predict_proba(row)[0, 1])
    expected_encoded = bundle['pipeline'].named_steps['preprocess'].transform(row)[0]
    np.testing.assert_allclose(local['encoded_values'], expected_encoded)


def test_local_explanation_accepts_numeric_strings_and_unknown_categories(trained_run):
    bundle = make_bundle(trained_run)
    local = local_explain(bundle, {'sensor': '0.2', 'type': 'unknown-category'})
    expected = bundle['pipeline'].predict_proba(pd.DataFrame({'sensor': [.2], 'type': ['unknown-category']}))[0, 1]
    assert local['probability'] == pytest.approx(expected)


@pytest.mark.parametrize('record', [
    {'sensor': 0.2},
    {'sensor': 0.2, 'type': 'H', 'extra': 1},
    {'sensor': [1, 2], 'type': 'H'},
    {'sensor': 'Infinity', 'type': 'H'},
    {'sensor': 'not a number', 'type': 'H'},
])
def test_local_explanation_rejects_invalid_record(trained_run, record):
    with pytest.raises(ValueError):
        local_explain(make_bundle(trained_run), record)


def test_bundle_roundtrip_uses_temporary_artifact(trained_run, monkeypatch, tmp_path):
    artifact = tmp_path / 'active_bundle.joblib'
    monkeypatch.setattr(core, 'ARTIFACT', artifact)
    run = trained_run
    stored = store_bundle(run['models']['XGBoost'], run['X_train'], ['sensor', 'type'], 'XGBoost')
    loaded = load_bundle()
    assert loaded['name'] == stored['name']
    assert loaded['id'] == stored['id']
    pd.testing.assert_frame_equal(loaded['background'], stored['background'])
    np.testing.assert_allclose(loaded['pipeline'].predict_proba(run['X_test']), run['models']['XGBoost'].predict_proba(run['X_test']))
    assert list(tmp_path.glob('*.tmp')) == []


def test_reactivation_creates_a_new_model_identity(trained_run, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'ARTIFACT', tmp_path / 'active_bundle.joblib')
    first = store_bundle(trained_run['models']['XGBoost'], trained_run['X_train'], ['sensor', 'type'])
    second = store_bundle(trained_run['models']['XGBoost'], trained_run['X_train'], ['sensor', 'type'])
    assert first['id'] != second['id']
    assert load_bundle()['id'] == second['id']


def test_failed_activation_preserves_existing_model(trained_run, monkeypatch, tmp_path):
    artifact = tmp_path / 'active_bundle.joblib'
    artifact.write_bytes(b'existing model preserved')
    monkeypatch.setattr(core, 'ARTIFACT', artifact)

    def failed_dump(bundle, destination):
        destination.write_bytes(b'partial write')
        raise OSError('Simulated storage failure')

    monkeypatch.setattr(core.joblib, 'dump', failed_dump)
    with pytest.raises(OSError, match='storage failure'):
        store_bundle(trained_run['models']['XGBoost'], trained_run['X_train'], ['sensor', 'type'])
    assert artifact.read_bytes() == b'existing model preserved'
    assert list(tmp_path.glob('*.tmp')) == []


def test_pdf_one_page():
    from io import BytesIO
    from pypdf import PdfReader
    content = create_onepager('Random Forest', {'AUC-ROC':.81,'PR-AUC (AP)':.52,
                              'Précision':.35,'Rappel':.72,'F1':.47}, n=1200,
                              positive_rate=.06, top_variables=['Tool wear [min]'])
    reader = PdfReader(BytesIO(content))
    assert len(reader.pages) == 1
    assert '0.81' in reader.pages[0].extract_text()
