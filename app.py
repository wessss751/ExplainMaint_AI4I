"""Streamlit application implementing all six modules of Challenge 1."""
from __future__ import annotations
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from core import (ARTIFACT, DEFAULT_FEATURES, IDENTIFIERS, LEAKAGE, TARGET,
                  candidate_estimators, global_explain, load_uci, local_explain,
                  model_importance, permutation_table, prepare_binary, store_bundle,
                  threshold_metrics, train_compare)
from report import create_onepager

st.set_page_config(page_title='ExplainMaint | ENSI', page_icon='⚙️', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.45rem; max-width: 1300px;}
h1 {color: #173B50; letter-spacing: -.5px;}
h2,h3 {color: #127F80;}
[data-testid="stMetric"] {background:#ECF4F5;border-radius:12px;padding:12px;}
</style>''', unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner='Téléchargement automatique UCI...')
def uci_data():
    return load_uci()

API_URL = os.environ.get('EXPLAINMAINT_API_URL', 'http://127.0.0.1:8000').rstrip('/')

def try_api(method, route, **kwargs):
    try:
        resp = requests.request(method, API_URL + route, timeout=75, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as exc:
        return None, str(exc)


def state_data():
    return st.session_state.get('dataset')


def expected_context(df):
    default = TARGET if TARGET in df else df.columns[-1]
    target = st.session_state.get('config_target', default)
    if target not in df:
        target = default
    return target


def need_result():
    obj = st.session_state.get('run')
    if obj is None:
        st.info('Lancez d’abord un entraînement dans « 3 · Modèles ».')
    return obj


def active_bundle(name, obj, features):
    bundle = store_bundle(obj['models'][name], obj['X_train'], features, name)
    st.session_state['active_name'] = name
    st.session_state['active_model_id'] = bundle['id']
    st.session_state.pop('global_cache', None)
    return bundle

st.sidebar.title('⚙️ ExplainMaint')
st.sidebar.caption('ENSI · Challenge 1 · Data Mining')
page = st.sidebar.radio('Parcours', ['1 · Données', '2 · Préparation', '3 · Modèles',
                                    '4 · Évaluation', '5 · XAI', '6 · API', 'Synthèse'], label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('AI4I 2020 • Classification binaire • SHAP + LIME')
st.title('ExplainMaint')
st.caption('Application de maintenance prédictive explicable | Démonstrateur pédagogique, données AI4I synthétiques')

# The data-source controls are always visible, independent of page, for easy recovery.
with st.sidebar.expander('Source des données', expanded=page.startswith('1')):
    source = st.radio('Choix', ['UCI · AI4I 2020', 'Importer CSV'], key='data_source')
    upload = st.file_uploader('CSV personnel (15 Mo max.)', type='csv') if source.startswith('Importer') else None
    sep = st.selectbox('Séparateur CSV', [',', ';', '\t'], index=0) if upload else ','
try:
    if source.startswith('UCI'):
        df = uci_data()
        dataset_key = 'uci_601'
    elif upload is not None:
        data_bytes = upload.getvalue()
        df = pd.read_csv(io.BytesIO(data_bytes), sep=sep)
        dataset_key = f'{upload.name}-{len(data_bytes)}-{hash(data_bytes)}-{sep}'
    else:
        df = None
        dataset_key = 'no_upload'
except Exception as exc:
    st.error(f'Chargement impossible : {exc}')
    df = None
    dataset_key = 'error'
if st.session_state.get('dataset_key') != dataset_key:
    st.session_state['dataset_key'] = dataset_key
    st.session_state.pop('run', None)
    st.session_state.pop('global_cache', None)
    st.session_state.pop('active_name', None)
    st.session_state.pop('active_model_id', None)
    for item in ('config_target','config_features','config_positive','config_test','training_signature','permutation_table','permutation_model','local_cache','onepager_pdf'):
        st.session_state.pop(item, None)
st.session_state['dataset'] = df
if df is None:
    st.warning('Chargez la source UCI ou importez un fichier CSV dans le panneau latéral pour commencer.')
    st.stop()
if df.empty or df.shape[1] < 2:
    st.error('Le CSV doit contenir au moins une observation et deux colonnes.')
    st.stop()

if page.startswith('1'):
    st.header('1 · Données et analyse exploratoire')
    a, b, c = st.columns(3)
    a.metric('Observations', f'{len(df):,}'.replace(',', ' '))
    b.metric('Variables', str(df.shape[1]))
    c.metric('Cellules manquantes', str(int(df.isna().sum().sum())))
    if source.startswith('UCI'):
        st.success('Données téléchargées automatiquement depuis UCI (ou repli) et mises en cache.')
        st.warning('Jeu synthétique : les cinq colonnes de modes de panne et les identifiants seront exclus des prédicteurs.')
    st.dataframe(df.head(25), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader('Statistiques descriptives')
        st.dataframe(df.describe(include='all').T.fillna('').astype(str), use_container_width=True)
    with c2:
        st.subheader('Valeurs manquantes')
        st.dataframe(df.isna().sum().rename('Manquantes').to_frame(), use_container_width=True)
    target = expected_context(df)
    st.subheader(f'Proportions de la cible · {target}')
    counts = df[target].astype(str).value_counts(dropna=False).reset_index()
    counts.columns = ['Classe', 'Effectif']
    st.plotly_chart(px.pie(counts, names='Classe', values='Effectif', hole=.45), use_container_width=True)
    st.subheader('Distributions et corrélations')
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    view = st.selectbox('Variable à explorer', df.columns.tolist())
    if view in numeric_cols:
        st.plotly_chart(px.histogram(df, x=view, nbins=35, color=df[target].astype(str),
                                     labels={'color': 'Classe'}, barmode='overlay'), use_container_width=True)
    else:
        vals = df.assign(**{view: df[view].astype(str)}).groupby([view, target], dropna=False).size().reset_index(name='Nombre')
        vals[target] = vals[target].astype(str)
        st.plotly_chart(px.bar(vals, x=view, y='Nombre', color=target, barmode='group'), use_container_width=True)
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1), use_container_width=True)
    if df[target].dropna().nunique() == 2:
        pos = '1' if '1' in df[target].astype(str).unique() else str(df[target].dropna().astype(str).value_counts().index[-1])
        chosen = st.selectbox('Taux de classe positive par variable', [x for x in df.columns if x != target])
        tmp = df[[chosen, target]].dropna().copy()
        tmp['_positive'] = (tmp[target].astype(str) == pos).astype(float)
        if pd.api.types.is_numeric_dtype(tmp[chosen]) and tmp[chosen].nunique() > 8:
            tmp['_bucket'] = pd.qcut(tmp[chosen], q=5, duplicates='drop').astype(str)
        else:
            tmp['_bucket'] = tmp[chosen].astype(str)
        rates = tmp.groupby('_bucket', as_index=False)['_positive'].mean()
        st.plotly_chart(px.bar(rates, x='_bucket', y='_positive', labels={'_positive':'Taux de classe 1', '_bucket':chosen}), use_container_width=True)

elif page.startswith('2'):
    st.header('2 · Préparation sans fuite de données')
    target = st.selectbox('Variable cible', df.columns.tolist(), index=df.columns.get_loc(expected_context(df)), key='w_target')
    classes = [str(v) for v in df[target].dropna().unique()]
    if len(classes) != 2:
        st.error(f'La variable sélectionnée a {len(classes)} classes. Choisir une cible binaire ou la binariser dans le CSV.')
        st.stop()
    positive_default = st.session_state.get('config_positive', '1' if '1' in classes else classes[-1])
    positive = st.selectbox('Classe positive (événement à anticiper)', classes, index=classes.index(positive_default) if positive_default in classes else 0, key='w_positive')
    if source.startswith('UCI'):
        forbidden = set(LEAKAGE + IDENTIFIERS + [target])
        valid = [c for c in df.columns if c not in forbidden]
        defaults = [c for c in DEFAULT_FEATURES if c in valid]
        st.info('Par sécurité, UDI, Product ID, TWF, HDF, PWF, OSF et RNF sont indisponibles comme variables explicatives.')
    else:
        valid = [c for c in df.columns if c != target]
        defaults = valid
        st.warning('Audit requis pour un CSV externe : ne pas utiliser de variable révélant un résultat connu après l’événement ou de variable ID.')
    features_default = st.session_state.get('config_features')
    if features_default is not None:
        defaults = [c for c in features_default if c in valid]
    features = st.multiselect('Variables explicatives', valid, default=defaults, key='w_features')
    test_size = st.slider('Part réservée au test (%)', 15, 40, st.session_state.get('config_test', 25), 5, key='w_test_percent')
    new_config = (target, str(positive), tuple(features), test_size)
    old_config = (st.session_state.get('config_target'), st.session_state.get('config_positive'),
                  tuple(st.session_state.get('config_features', [])), st.session_state.get('config_test'))
    if new_config != old_config:
        for item in ('run', 'active_name', 'active_model_id', 'global_cache', 'permutation_table', 'local_cache', 'onepager_pdf'):
            st.session_state.pop(item, None)
    st.session_state.update(config_target=target, config_positive=str(positive),
                            config_features=list(features), config_test=test_size)
    st.write('Imputation : médiane (numérique), mode (catégorielle) ; encodage one-hot ; StandardScaler sur les numériques. Ajustés sur train uniquement.')
    try:
        X, y = prepare_binary(df, target, features, positive)
        p, q, r = st.columns(3)
        p.metric('Variables retenues', len(features))
        q.metric('Classe positive', f'{y.mean():.2%}')
        r.metric('Taille test visée', f'{test_size}%')
        st.success('Configuration valide. Vous pouvez lancer les modèles. Toute modification de préparation invalide un entraînement antérieur.')
    except Exception as exc:
        st.error(str(exc))

elif page.startswith('3'):
    st.header('3 · Modélisation et comparaison')
    target = expected_context(df)
    chosen_features = st.session_state.get('config_features')
    if chosen_features is None:
        forbidden = set(LEAKAGE + IDENTIFIERS + [target]) if source.startswith('UCI') else {target}
        chosen_features = [c for c in df if c not in forbidden]
    positive = st.session_state.get('config_positive', '1' if '1' in df[target].astype(str).unique() else str(df[target].dropna().iloc[-1]))
    try:
        X, y = prepare_binary(df, target, chosen_features, positive)
    except Exception as exc:
        st.error('Vérifier l’onglet Préparation : ' + str(exc))
        st.stop()
    st.info(f'Configuration : {len(X)} lignes, {len(chosen_features)} variables, cible {target} = {positive}, classe positive {y.mean():.2%}.')
    names = list(candidate_estimators(y, {}).keys())
    chosen_models = st.multiselect('Modèles à entraîner', names, default=names)
    c1, c2, c3 = st.columns(3)
    with c1:
        depth = st.slider('Profondeur des arbres', 3, 10, 5)
        trees = st.slider('Nombre d’arbres (RF / XGB)', 60, 300, 130, 10)
    with c2:
        neighbors = st.slider('Voisins KNN', 3, 25, 7, 2)
        C = st.select_slider('Régularisation LR (C)', options=[.05,.1,.5,1.,3.,10.], value=1.)
    with c3:
        lr = st.select_slider('Learning rate XGBoost', options=[.03,.05,.07,.1,.15], value=.07)
        tune = st.checkbox('Optimiser les hyperparamètres (3 plis × 4 tirages / modèle) : plus lent', value=False)
    if st.button('🚀 Entraîner et comparer', type='primary', use_container_width=True):
        bar = st.progress(0, text='Entraînement en cours...')
        try:
            result = train_compare(X, y, chosen_models, test_size=st.session_state.get('config_test', 25)/100,
                                   params={'depth':depth,'trees':trees,'neighbors':neighbors,'C':C,'lr':lr}, tune=tune,
                                   progress=lambda v, name: bar.progress(v, text=f'Terminé : {name}'))
            st.session_state['run'] = result
            st.session_state['training_signature'] = (dataset_key, target, str(positive), tuple(chosen_features),
                                                       st.session_state.get('config_test', 25))
            for item in ('active_name', 'active_model_id', 'global_cache', 'permutation_table', 'permutation_model', 'local_cache', 'onepager_pdf'):
                st.session_state.pop(item, None)
            bar.empty()
            st.success('Entraînement terminé. Résultats calculés sur le test isolé.')
        except Exception as exc:
            bar.empty()
            st.exception(exc)
    if st.session_state.get('run') is not None:
        result = st.session_state['run']
        st.subheader('Tableau comparatif (seuil = 0,50)')
        score_df = pd.DataFrame(result['scores']).T.sort_values('PR-AUC (AP)', ascending=False)
        st.dataframe(score_df.style.format('{:.4f}'), use_container_width=True)
        st.download_button('Exporter les métriques CSV', score_df.to_csv().encode('utf-8-sig'), 'comparaison_modeles.csv', 'text/csv')
        if result['tuned']:
            st.json(result['tuned'])
        st.caption('Une bonne AUC-ROC peut masquer des performances modestes en classe rare : examiner AP et rappel.')

elif page.startswith('4'):
    st.header('4 · Évaluation interactive')
    result = need_result()
    if result:
        y_te = result['y_test']
        name = st.selectbox('Modèle détaillé', list(result['models']))
        threshold = st.slider('Seuil de décision (probabilité de panne)', 0., 1., .5, .01, key='decision_threshold')
        scores = threshold_metrics(y_te, result['probs'][name], threshold)
        cols = st.columns(5)
        for col, (metric, val) in zip(cols, scores.items()):
            col.metric(metric, f'{val:.3f}')
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for label, prob in result['probs'].items():
                fpr, tpr, _ = roc_curve(y_te, prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=label, mode='lines'))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Aléatoire', mode='lines', line={'dash':'dash','color':'gray'}))
            fig.update_layout(title='Courbes ROC superposées', xaxis_title='Taux faux positifs', yaxis_title='Rappel (TPR)', height=390)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            for label, prob in result['probs'].items():
                precision, recall, _ = precision_recall_curve(y_te, prob)
                fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=label))
            fig.update_layout(title='Courbes précision-rappel', xaxis_title='Rappel', yaxis_title='Précision', height=390)
            st.plotly_chart(fig, use_container_width=True)
        pred = (result['probs'][name] >= threshold).astype(int)
        matrix = confusion_matrix(y_te, pred, labels=[0,1])
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = px.imshow(matrix, text_auto=True, x=['Prédit : 0','Prédit : 1'],
                            y=['Réel : 0','Réel : 1'], color_continuous_scale='Teal', title='Matrice de confusion')
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.metric('Pannes manquées (FN)', int(matrix[1,0]))
            st.metric('Fausses alertes (FP)', int(matrix[0,1]))
            st.write('Réduire le seuil peut augmenter le rappel mais provoquer davantage de fausses alertes. Choisir selon les coûts opérationnels.')

elif page.startswith('5'):
    st.header('5 · Explainable AI : global et individuel')
    result = need_result()
    if result:
        name = st.selectbox('Modèle à interpréter', list(result['models']), key='xai_model')
        if st.button('Activer ce modèle dans l’API FastAPI', type='primary'):
            feature_list = list(result['X_train'].columns)
            active_bundle(name, result, feature_list)
            st.success(f'Activé : {name} → {ARTIFACT}. L’application et l’API partagent le même modèle.')
        if st.session_state.get('active_name') != name:
            st.info('Pour exploiter l’API, cliquez sur « Activer ce modèle ». Les explications ci-dessous peuvent aussi être calculées localement.')
        bundle = {'pipeline':result['models'][name],
                  'background':result['X_train'].sample(min(120,len(result['X_train'])),random_state=42),
                  'features':list(result['X_train'].columns), 'name':name}
        st.subheader('Explications globales')
        native = model_importance(bundle)
        if not native.empty:
            st.plotly_chart(px.bar(native.head(18).sort_values('Importance'), x='Importance', y='Variable', orientation='h',
                                    title='Importance native / coefficients du modèle (variables encodées)'), use_container_width=True)
        else:
            st.caption('Ce modèle ne fournit pas de coefficients ou importances natives. L’importance par permutation fonctionne pour tous.')
        if st.button('Calculer importance par permutation sur test'):
            with st.spinner('Perturbation des variables et comparaison de la performance...'):
                perm = permutation_table(bundle, result['X_test'], result['y_test'])
                st.session_state['permutation_table'] = perm
                st.session_state['permutation_model'] = name
        if st.session_state.get('permutation_table') is not None and st.session_state.get('permutation_model') == name:
            perm = st.session_state['permutation_table']
            st.plotly_chart(px.bar(perm.sort_values('Importance AP'), x='Importance AP', y='Variable', orientation='h',
                                    title='Chute de AP par permutation (variables originales)'), use_container_width=True)
        if st.button('Calculer SHAP global (échantillon de 25 observations)'):
            with st.spinner('Calcul SHAP par permutations, quelques instants...'):
                if st.session_state.get('active_name') == name:
                    glob, err = try_api('GET', '/global', params={
                        'n': 25, 'expected_model_id': st.session_state.get('active_model_id')})
                else:
                    glob, err = None, 'Modèle non activé'
                if glob is None:
                    st.caption('API indisponible ou autre modèle : calcul local de secours. ' + str(err)[:110])
                    glob = global_explain(bundle, n=25)
                else:
                    st.success('SHAP global reçu depuis FastAPI (/global).')
                st.session_state['global_cache'] = (name, glob)
        cached = st.session_state.get('global_cache')
        if cached and cached[0] == name:
            glob = cached[1]
            ex = shap.Explanation(values=np.array(glob['values']), base_values=np.array(glob['base_values']),
                                  data=np.array(glob['data']), feature_names=glob['names'])
            try:
                plt.close('all')
                shap.plots.beeswarm(ex, show=False, max_display=min(12,len(glob['names'])))
                st.pyplot(plt.gcf(), clear_figure=True)
                idx = st.selectbox('Variable pour la dépendance SHAP', range(len(glob['names'])),
                                   format_func=lambda i: glob['names'][i])
                plt.close('all')
                shap.plots.scatter(ex[:, idx], show=False)
                st.pyplot(plt.gcf(), clear_figure=True)
            except Exception as exc:
                st.warning('Tracé SHAP indisponible : ' + str(exc))
            st.caption('SHAP par permutations sur probabilité de panne ; variables catégorielles représentées après one-hot encoding.')
        st.divider()
        st.subheader('Explication locale : même individu pour SHAP et LIME')
        approach = st.radio('Source de l’individu', ['Sélectionner un individu du test', 'Saisir un individu fictif'], horizontal=True)
        if approach.startswith('Sélectionner'):
            idx = st.selectbox('Ligne du jeu de test', range(len(result['X_test'])), format_func=lambda i: f'Individu #{i} | classe réelle : {int(result["y_test"].iloc[i])}')
            record = result['X_test'].iloc[idx].to_dict()
            st.dataframe(pd.DataFrame([record]), use_container_width=True)
        else:
            record = {}
            with st.form('local_input'):
                cols = st.columns(2)
                for i, feature in enumerate(bundle['features']):
                    series = result['X_train'][feature]
                    if pd.api.types.is_numeric_dtype(series):
                        record[feature] = cols[i%2].number_input(feature, value=float(series.median()), format='%.3f')
                    else:
                        choices = sorted(series.dropna().astype(str).unique().tolist())
                        record[feature] = cols[i%2].selectbox(feature, choices)
                st.form_submit_button('Valider les valeurs')
        th = st.session_state.get('decision_threshold', .5)
        if st.button('Expliquer cet individu (SHAP + LIME)', type='primary'):
            with st.spinner('Calcul de deux explications locales...'):
                if st.session_state.get('active_name') == name:
                    local, err = try_api('POST', '/explain', json={
                        'record': record, 'threshold': th,
                        'expected_model_id': st.session_state.get('active_model_id')})
                else:
                    local, err = None, 'Modèle non activé'
                if local is None:
                    st.caption('Mode de secours : explication locale calculée dans Streamlit. ' + str(err)[:120])
                    try:
                        local = local_explain(bundle, record)
                        local['decision'] = int(local['probability'] >= th)
                    except Exception as exc:
                        st.error('Impossible de calculer XAI : ' + str(exc))
                else:
                    st.success('Réponse reçue de FastAPI (/explain).')
                if local:
                    st.session_state['local_cache'] = (name, record, local)
        loc = st.session_state.get('local_cache')
        if (loc and loc[0] == name
                and pd.Series(loc[1]).equals(pd.Series(record))):
            vals = loc[2]
            st.metric('Probabilité estimée de panne', f'{vals["probability"]:.2%}')
            st.write('**Décision au seuil choisi :** ' + ('Panne potentielle : inspecter' if vals['probability'] >= th else 'Fonctionnement prédit'))
            st.write(vals['summary'])
            l1,l2 = st.columns(2)
            with l1:
                st.markdown('**SHAP waterfall**')
                plt.close('all')
                ex = shap.Explanation(values=np.array(vals['shap_values']),
                                      base_values=vals['base_value'], data=np.array(vals['encoded_values']),
                                      feature_names=vals['names'])
                try:
                    shap.plots.waterfall(ex, show=False, max_display=min(12,len(vals['names'])))
                    st.pyplot(plt.gcf(), clear_figure=True)
                except Exception as exc:
                    st.warning(str(exc))
            with l2:
                st.markdown('**LIME (même observation)**')
                lime_data = pd.DataFrame(vals['lime'])
                st.plotly_chart(px.bar(lime_data.iloc[::-1], x='weight', y='variable', orientation='h',
                                        title='Coefficients locaux LIME'), use_container_width=True)
            st.caption('SHAP et LIME ne sont pas nécessairement identiques : SHAP répartit les contributions selon un fond de référence ; LIME ajuste une approximation locale sur des perturbations.')

elif page.startswith('6'):
    st.header('6 · API d’interprétation (FastAPI)')
    st.code('uvicorn api:app --host 127.0.0.1 --port 8000', language='bash')
    st.write('Appels serveur → serveur ; URL de configuration : ', API_URL)
    check, error = try_api('GET', '/health')
    if check:
        st.success(f'API connectée · Modèle disponible : {check["trained_model_available"]}')
        model, _ = try_api('GET', '/model')
        if model: st.json(model)
    else:
        st.warning('API non joignable. Démarrez le service ou déployez le Dockerfile. Détail : ' + str(error)[:180])
    st.markdown('Endpoints : `GET /health` · `GET /model` · `GET /global?n=25` · `POST /explain` · `GET /docs`.')
    if os.environ.get('EXPLAINMAINT_API_ROOT_PATH') == '/api':
        st.markdown('[Documentation publique FastAPI](/api/docs) · [État du service](/api/health)')
    st.code('''POST http://localhost:8000/explain
Content-Type: application/json
{"threshold": 0.5, "record": {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1500, "Torque [Nm]": 40.0, "Tool wear [min]": 150}}''', language='http')
    st.caption('Le corps doit refléter les variables du modèle actif. OpenAPI documente les formats et réponses. La route /explain exécute réellement SHAP et LIME.')

else:
    st.header('Synthèse et PDF de dépôt')
    result = need_result()
    if result:
        metrics = pd.DataFrame(result['scores']).T.sort_values('PR-AUC (AP)',ascending=False)
        st.dataframe(metrics, use_container_width=True)
        default_name = metrics.index[0]
        name = st.selectbox('Modèle à retenir pour le rapport', metrics.index.tolist(), index=0)
        threshold = st.slider('Seuil retenu pour le one-pager', 0., 1., st.session_state.get('decision_threshold', .5), .01)
        measured = threshold_metrics(result['y_test'], result['probs'][name], threshold)
        st.info('Le modèle suggéré en premier maximise AP sur le test : choix descriptif, non une sélection validée par jeu indépendant. À discuter selon le métier.')
        variables = []
        cached = st.session_state.get('global_cache')
        if cached and cached[0] == name:
            glob = cached[1]
            avg = np.abs(np.array(glob['values'])).mean(axis=0)
            variables = [glob['names'][i] for i in np.argsort(-avg)[:4]]
        report_context = (name, threshold, tuple(variables))
        if st.session_state.get('onepager_context') != report_context:
            st.session_state.pop('onepager_pdf', None)
        if st.button('Générer le one-pager PDF avec les résultats calculés', type='primary'):
            pdf = create_onepager(name, measured, len(result['X_train']) + len(result['X_test']),
                                  float(pd.concat([result['y_train'], result['y_test']]).mean()), threshold,
                                  variables, custom_dataset=not source.startswith('UCI'))
            st.session_state['onepager_pdf'] = pdf
            st.session_state['onepager_context'] = report_context
        if st.session_state.get('onepager_pdf'):
            st.download_button('📄 Télécharger le PDF final (une page)', st.session_state['onepager_pdf'],
                               'one_pager_resultats_mesures.pdf', 'application/pdf')
    st.markdown('**Discussion attendue :** cohérence SHAP / LIME et intuition métier ; variables potentiellement sensibles ; arbitrage performance / lisibilité ; actions d’inspection ; limites des données synthétiques et de la validation aléatoire.')
