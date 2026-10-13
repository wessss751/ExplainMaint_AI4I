"""ExplainMaint: data, modelling and post-hoc interpretability.

No transformations are fitted before the train/test split. Positive class == 1.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED = 42
DATA_URL = 'https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip'
MIRROR_CSV = 'https://raw.githubusercontent.com/jackquill/ai4i-predictive-maintenance/main/data/ai4i2020.csv'
TARGET = 'Machine failure'
LEAKAGE = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
IDENTIFIERS = ['UDI', 'UID', 'Product ID']
DEFAULT_FEATURES = ['Type', 'Air temperature [K]', 'Process temperature [K]',
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
ARTIFACT = Path(os.environ.get('MODEL_PATH', 'artifacts/active_bundle.joblib'))


def load_uci() -> pd.DataFrame:
    """Download source automatically; retain a cached local copy for offline reruns."""
    cache = Path('data/ai4i2020.csv')
    if cache.exists():
        return pd.read_csv(cache, encoding='utf-8-sig')
    errors = []
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=601)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        # The ucimlrepo API may expose the outcome subtypes as target columns.
        if TARGET not in df and 'Machine failure' in ds.data.original:
            df = ds.data.original.copy()
        if TARGET not in df:
            raise ValueError('La cible Machine failure est absente du téléchargement UCI.')
    except Exception as exc:
        errors.append(str(exc))
        try:
            import requests
            import zipfile
            resp = requests.get(DATA_URL, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = next(n for n in zf.namelist() if n.lower().endswith('ai4i2020.csv'))
                df = pd.read_csv(zf.open(name), encoding='utf-8-sig')
        except Exception as exc2:
            errors.append(str(exc2))
            try:
                import requests
                resp = requests.get(MIRROR_CSV, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(io.BytesIO(resp.content), encoding='utf-8-sig')
            except Exception as exc3:
                errors.append(str(exc3))
                raise RuntimeError('Téléchargement impossible. Vérifier internet ou importer le CSV. ' + ' | '.join(errors)) from exc3
    # ucimlrepo names sometimes omit units; standardise to the source CSV schema.
    df = df.rename(columns={'Air temperature': 'Air temperature [K]',
                            'Process temperature': 'Process temperature [K]',
                            'Rotational speed': 'Rotational speed [rpm]',
                            'Torque': 'Torque [Nm]', 'Tool wear': 'Tool wear [min]'})
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def prepare_binary(df: pd.DataFrame, target: str, features: list[str], positive_value: str):
    if target not in df:
        raise ValueError('Cible absente.')
    if (not features or target in features or len(features) != len(set(features))
            or any(c not in df for c in features)):
        raise ValueError('Sélection des variables invalide.')
    if target == TARGET:
        forbidden = sorted(set(features).intersection(LEAKAGE + IDENTIFIERS))
        if forbidden:
            raise ValueError('Variables exclues pour éviter la fuite de cible AI4I : ' + ', '.join(forbidden))
    X = df[features].copy().replace([np.inf, -np.inf], np.nan)
    yraw = df[target]
    valid = yraw.notna()
    X = X.loc[valid]
    labels = yraw.loc[valid].astype(str)
    if labels.nunique() != 2:
        raise ValueError('La cible doit contenir exactement deux classes.')
    if str(positive_value) not in labels.unique():
        raise ValueError('La classe positive choisie est absente de la cible.')
    y = (labels == str(positive_value)).astype(int)
    if y.nunique() != 2 or y.value_counts().min() < 3:
        raise ValueError('La cible doit contenir deux classes dont au moins 3 individus dans chaque classe.')
    for name in X.columns:
        if X[name].dtype == 'object':
            X[name] = X[name].replace(r'^\s*$', np.nan, regex=True)
    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError('Variables entièrement vides à retirer : ' + ', '.join(all_missing))
    return X, y


def make_preprocessor(X: pd.DataFrame):
    numerics = X.select_dtypes(include=np.number).columns.tolist()
    categoricals = [c for c in X if c not in numerics]
    transformers = []
    if numerics:
        transformers.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                                              ('scale', StandardScaler())]), numerics))
    if categoricals:
        transformers.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                              ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categoricals))
    return ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)


def candidate_estimators(y_train: pd.Series, params: dict):
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    depth = int(params.get('depth', 5))
    trees = int(params.get('trees', 130))
    neighbors = int(params.get('neighbors', 7))
    return {
        'Régression logistique': LogisticRegression(max_iter=1500, class_weight='balanced', C=float(params.get('C', 1))),
        'Arbre de décision': DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, class_weight='balanced', random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=neighbors, weights='distance'),
        'Random Forest': RandomForestClassifier(n_estimators=trees, max_depth=depth + 3, min_samples_leaf=2,
                                                class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=trees, max_depth=depth, learning_rate=float(params.get('lr', .07)),
                                  subsample=.9, colsample_bytree=.9, scale_pos_weight=ratio,
                                  tree_method='hist', eval_metric='logloss', objective='binary:logistic',
                                  n_jobs=2, random_state=SEED),
    }

SEARCH_SPACE = {
    'Régression logistique': {'model__C': [.05, .1, .5, 1, 3, 10]},
    'Arbre de décision': {'model__max_depth': [3, 4, 5, 7, 9], 'model__min_samples_leaf': [2, 4, 8, 15]},
    'KNN': {'model__n_neighbors': [3, 5, 7, 11, 15], 'model__weights': ['uniform', 'distance']},
    'Random Forest': {'model__max_depth': [4, 6, 8, 12, None], 'model__min_samples_leaf': [1, 2, 4]},
    'XGBoost': {'model__max_depth': [3, 4, 5, 6], 'model__learning_rate': [.03, .05, .1], 'model__subsample': [.8, 1.]},
}


def threshold_metrics(y, prob, threshold=.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {'AUC-ROC': round(float(roc_auc_score(y, prob)), 4),
            'PR-AUC (AP)': round(float(average_precision_score(y, prob)), 4),
            'Précision': round(float(precision_score(y, pred, zero_division=0)), 4),
            'Rappel': round(float(recall_score(y, pred, zero_division=0)), 4),
            'F1': round(float(f1_score(y, pred, zero_division=0)), 4)}


def train_compare(X, y, model_names, test_size=.25, params=None, tune=False, progress=None):
    if not model_names:
        raise ValueError('Sélectionner au moins un modèle.')
    params = params or {}
    if len(y) < 30:
        raise ValueError('Au moins 30 observations sont recommandées.')
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=float(test_size), random_state=SEED, stratify=y)
    if y_tr.value_counts().min() < 3 or y_te.nunique() != 2:
        raise ValueError('Classes insuffisamment représentées dans le découpage ; augmenter la taille du jeu.')
    available = candidate_estimators(y_tr, params)
    result = {'models': {}, 'probs': {}, 'scores': {}, 'X_train': X_tr, 'X_test': X_te,
              'y_train': y_tr, 'y_test': y_te, 'tuned': {}, 'test_size': test_size}
    for i, name in enumerate(model_names):
        if name not in available:
            raise ValueError(f'Modèle inconnu : {name}')
        pipe = Pipeline([('preprocess', make_preprocessor(X_tr)), ('model', available[name])])
        if tune:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            search = RandomizedSearchCV(pipe, SEARCH_SPACE[name], n_iter=4, cv=cv, n_jobs=1,
                                        scoring='average_precision' if y_tr.mean() < .15 else 'roc_auc',
                                        random_state=SEED, error_score='raise', refit=True)
            search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
            result['tuned'][name] = search.best_params_
        else:
            pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        result['models'][name] = pipe
        result['probs'][name] = prob
        result['scores'][name] = threshold_metrics(y_te, prob)
        if progress:
            progress((i + 1) / len(model_names), name)
    return result


def store_bundle(model, X_train, selected_features, model_name='Modèle'):
    artifact = ARTIFACT
    artifact.parent.mkdir(parents=True, exist_ok=True)
    sample = X_train.sample(n=min(len(X_train), 120), random_state=SEED).copy()
    bundle = {'pipeline': model, 'background': sample, 'features': list(selected_features),
              'name': model_name, 'id': uuid4().hex}
    # A unique temporary file avoids collisions between simultaneous activations.
    with tempfile.NamedTemporaryFile(dir=artifact.parent, suffix='.tmp', delete=False) as handle:
        temp = Path(handle.name)
    try:
        joblib.dump(bundle, temp)
        os.replace(temp, artifact)
    finally:
        temp.unlink(missing_ok=True)
    return bundle


def load_bundle():
    if not ARTIFACT.exists():
        raise FileNotFoundError('Aucun modèle actif. Entraîner puis activer un modèle dans Streamlit.')
    return joblib.load(ARTIFACT)


def transformed(bundle, rows):
    pipe = bundle['pipeline']
    data = pipe.named_steps['preprocess'].transform(rows)
    names = pipe.named_steps['preprocess'].get_feature_names_out()
    return np.asarray(data, dtype=float), list(names)


def model_importance(bundle):
    pipe = bundle['pipeline']
    model = pipe.named_steps['model']
    _, names = transformed(bundle, bundle['background'].head(1))
    values = None
    if hasattr(model, 'coef_'):
        values = np.ravel(model.coef_)
    elif hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
    if values is None:
        return pd.DataFrame(columns=['Variable', 'Importance'])
    return pd.DataFrame({'Variable': names, 'Importance': values}).sort_values('Importance', key=abs, ascending=False)


def permutation_table(bundle, X_test, y_test, sample_size=700):
    if len(X_test) > sample_size:
        _, X_part, _, y_part = train_test_split(X_test, y_test, test_size=sample_size/len(X_test),
                                                random_state=SEED, stratify=y_test)
    else:
        X_part, y_part = X_test, y_test
    res = permutation_importance(bundle['pipeline'], X_part, y_part,
                                 scoring='average_precision', n_repeats=3, random_state=SEED, n_jobs=1)
    return pd.DataFrame({'Variable': X_part.columns, 'Importance AP': res.importances_mean,
                         'Ecart-type': res.importances_std}).sort_values('Importance AP', ascending=False)


def _shap_explanation(bundle, rows):
    import shap
    bg, names = transformed(bundle, bundle['background'].head(55))
    foreground, _ = transformed(bundle, rows)
    clf = bundle['pipeline'].named_steps['model']
    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(lambda z: clf.predict_proba(z)[:, 1], masker,
                               algorithm='permutation', feature_names=names, seed=SEED)
    explanation = explainer(foreground, max_evals=max(2 * len(names) + 1, 21))
    return explanation, foreground, names


def global_explain(bundle, n=25):
    sample = bundle['background'].head(min(n, len(bundle['background'])))
    explanation, values, names = _shap_explanation(bundle, sample)
    return {'names': names, 'values': np.asarray(explanation.values).tolist(),
            'base_values': np.asarray(explanation.base_values).tolist(),
            'data': values.tolist()}


def local_explain(bundle, record: dict):
    from lime.lime_tabular import LimeTabularExplainer
    missing = sorted(set(bundle['features']) - set(record))
    extra = sorted(set(record) - set(bundle['features']))
    if missing or extra:
        raise ValueError(f'Variables manquantes : {missing} ; variables supplémentaires : {extra}')
    for name, value in record.items():
        if not pd.api.types.is_scalar(value):
            raise ValueError(f'La variable {name} doit contenir une valeur simple ou null.')
    row = pd.DataFrame([{c: record[c] for c in bundle['features']}], columns=bundle['features'])
    # Restore expected dtypes; prevents string JSON numeric values being encoded as categories.
    for c, dtype in bundle['background'].dtypes.items():
        if pd.api.types.is_numeric_dtype(dtype):
            row[c] = pd.to_numeric(row[c], errors='raise')
            if np.isinf(row[c].to_numpy(dtype=float)).any():
                raise ValueError(f'La variable {c} doit être finie ou null.')
        else:
            # Keep missing values missing so the fitted modal imputer handles them.
            row[c] = row[c].replace(r'^\s*$', np.nan, regex=True)
            row[c] = row[c].where(row[c].notna(), np.nan)
    explanation, encoded, names = _shap_explanation(bundle, row)
    train_encoded, _ = transformed(bundle, bundle['background'])
    clf = bundle['pipeline'].named_steps['model']
    lime_exp = LimeTabularExplainer(train_encoded, feature_names=names,
                                    class_names=['Fonctionnement', 'Panne'], mode='classification',
                                    discretize_continuous=False, random_state=SEED)
    lime_result = lime_exp.explain_instance(encoded[0], clf.predict_proba,
                                            num_features=len(names), num_samples=1200)
    prob = float(bundle['pipeline'].predict_proba(row)[0, 1])
    shap_values = np.asarray(explanation.values[0], dtype=float)
    order = np.argsort(-np.abs(shap_values))
    top = [{'variable': names[i], 'contribution': round(float(shap_values[i]), 5)} for i in order[:3]]
    names_positive = [t['variable'].replace('num__', '').replace('cat__', '') for t in top if t['contribution'] > 0]
    phrase = ('Le modèle estime un risque de panne de ' + f'{prob:.1%}. ' +
              ('Les facteurs qui augmentent ce risque sont : ' + ', '.join(names_positive[:3]) + '.'
               if names_positive else 'Les variables dominantes ne contribuent pas positivement au risque.'))
    return {'model': bundle['name'], 'probability': prob, 'base_value': float(np.asarray(explanation.base_values).ravel()[0]),
            'names': names, 'encoded_values': encoded[0].tolist(), 'shap_values': shap_values.tolist(),
            'lime': [{'variable': str(name), 'weight': float(weight)} for name, weight in lime_result.as_list(label=1)],
            'summary': phrase, 'top_shap': top}
