"""Exercise the real Streamlit UI and save screenshots from a running Docker app.

Optional tool: install Playwright, then use an installed Edge browser (default).
This trains the five models and activates XGBoost in the supplied application.
It never publishes files or modifies the original submission PDF/captures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


MODEL_NAMES = ['Régression logistique', 'Arbre de décision', 'KNN', 'Random Forest', 'XGBoost']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_url', help='Running application URL, e.g. http://127.0.0.1:7860')
    parser.add_argument('--output', type=Path, required=True, help='New screenshots directory')
    parser.add_argument('--browser-channel', default='msedge', help='Installed Chromium channel; default: msedge')
    parser.add_argument('--browser-executable', default=os.environ.get('EXPLAINMAINT_BROWSER_EXECUTABLE'),
                        help='Explicit browser executable; overrides --browser-channel')
    parser.add_argument('--timeout-ms', type=int, default=180000)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    url = urlsplit(base)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error('Use an HTTP(S) application URL without credentials, query or fragment')
    try:
        from playwright.sync_api import expect, sync_playwright
    except ImportError:
        parser.error('Optional dependency missing: install playwright to run this browser check')

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output.parent / 'browser_validation.json'
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'base_url': base,
              'passed': False, 'steps': [], 'artifacts': [], 'browser_errors': []}

    def artifact(path):
        report['artifacts'].append({'path': path.as_posix(), 'bytes': path.stat().st_size,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    def step(name):
        report['steps'].append({'name': name, 'status': 'passed'})
        print(f'PASS {name}', flush=True)

    def save_report():
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    with sync_playwright() as playwright:
        browser = None
        page = None
        try:
            options = {'headless': True}
            if args.browser_executable:
                options['executable_path'] = args.browser_executable
            else:
                options['channel'] = args.browser_channel
            browser = playwright.chromium.launch(**options)
            report['browser_version'] = browser.version
            context = browser.new_context(viewport={'width': 1512, 'height': 1200},
                                          device_scale_factor=1, locale='fr-FR', accept_downloads=True)
            context.set_default_timeout(args.timeout_ms)
            page = context.new_page()
            page.on('pageerror', lambda error: report['browser_errors'].append(str(error)))
            expect.set_options(timeout=args.timeout_ms)

            def clean_page():
                expect(page.get_by_test_id('stException')).to_have_count(0)
                expect(page.get_by_text(re.compile(r'Chargement impossible|Impossible de calculer XAI|Tracé SHAP indisponible'))).to_have_count(0)

            def capture(name, focus=None, target=None):
                selected = target or page
                if focus is not None:
                    focus.scroll_into_view_if_needed()
                    focus.evaluate("element => element.scrollIntoView({block: 'start', inline: 'nearest'})")
                selected.evaluate('document.fonts.ready')
                destination = args.output / name
                selected.screenshot(path=str(destination), animations='disabled')
                artifact(destination)

            def navigate(label, heading):
                page.get_by_test_id('stSidebar').get_by_text(label, exact=True).click()
                expect(page.get_by_role('heading', name=heading, exact=True)).to_be_visible()
                clean_page()

            def select_model(label):
                selector = page.get_by_test_id('stSelectbox').filter(has=page.get_by_text(label, exact=True))
                selector.get_by_role('combobox').click()
                page.get_by_role('option', name='XGBoost', exact=True).click()
                expect(selector).to_contain_text('XGBoost')

            page.goto(base, wait_until='domcontentloaded')
            expect(page.get_by_role('heading', name='1 · Données et analyse exploratoire', exact=True)).to_be_visible()
            observation_metric = page.get_by_test_id('stMetric').filter(has=page.get_by_text('Observations', exact=True))
            expect(observation_metric.get_by_test_id('stMetricValue')).to_have_text('10 000')
            expect(page.get_by_text('Données téléchargées automatiquement depuis UCI (ou repli) et mises en cache.', exact=True)).to_be_visible()
            clean_page()
            capture('01_donnees_navigateur.png', page.get_by_role('heading', name='1 · Données et analyse exploratoire', exact=True))
            step('UCI loaded: 10000 observations')

            navigate('2 · Préparation', '2 · Préparation sans fuite de données')
            expect(page.get_by_text('Configuration valide. Vous pouvez lancer les modèles. Toute modification de préparation invalide un entraînement antérieur.', exact=True)).to_be_visible()
            capture('02_preparation_navigateur.png', page.get_by_role('heading', name='2 · Préparation sans fuite de données', exact=True))
            step('Preparation: default AI4I predictors and stratified test settings')

            navigate('3 · Modèles', '3 · Modélisation et comparaison')
            model_selector = page.get_by_test_id('stMultiSelect').filter(has=page.get_by_text('Modèles à entraîner', exact=True))
            for name in MODEL_NAMES:
                expect(model_selector).to_contain_text(name)
            page.get_by_role('button', name='🚀 Entraîner et comparer', exact=True).click()
            expect(page.get_by_text('Entraînement terminé. Résultats calculés sur le test isolé.', exact=True)).to_be_visible()
            clean_page()
            with page.expect_download() as download_event:
                page.get_by_role('button', name='Exporter les métriques CSV', exact=True).click()
            csv_path = args.output.parent / 'comparaison_modeles_navigateur.csv'
            download_event.value.save_as(str(csv_path))
            with csv_path.open(encoding='utf-8-sig', newline='') as handle:
                csv_rows = list(csv.reader(handle))
            if len(csv_rows) != 6 or {row[0] for row in csv_rows[1:]} != set(MODEL_NAMES):
                raise AssertionError('The downloaded comparison must contain exactly the five trained models')
            artifact(csv_path)
            capture('03_modeles_navigateur.png', page.get_by_role('heading', name='Tableau comparatif (seuil = 0,50)', exact=True))
            step('All five models trained through the UI; actual metrics CSV downloaded')

            navigate('4 · Évaluation', '4 · Évaluation interactive')
            select_model('Modèle détaillé')
            expect(page.get_by_test_id('stPlotlyChart')).to_have_count(3)
            capture('04_evaluation_xgboost_navigateur.png', page.get_by_role('heading', name='4 · Évaluation interactive', exact=True))
            step('XGBoost evaluation and comparison charts')

            navigate('5 · XAI', '5 · Explainable AI : global et individuel')
            select_model('Modèle à interpréter')
            page.get_by_role('button', name='Activer ce modèle dans l’API FastAPI', exact=True).click()
            expect(page.get_by_text(re.compile(r'^Activé : XGBoost'))).to_be_visible()
            model_response = context.request.get(base + '/api/model')
            if model_response.status != 200 or model_response.json().get('name') != 'XGBoost':
                raise AssertionError('The public FastAPI model does not match the activated XGBoost model')
            report['active_model'] = model_response.json()
            step('XGBoost activated through UI and confirmed by public FastAPI')

            page.get_by_role('button', name='Calculer SHAP global (échantillon de 25 observations)', exact=True).click()
            global_success = page.get_by_text('SHAP global reçu depuis FastAPI (/global).', exact=True)
            expect(global_success).to_be_visible()
            expect(page.get_by_test_id('stImage')).to_have_count(2)
            clean_page()
            capture('05_shap_global_fastapi_navigateur.png', global_success)
            step('Global SHAP returned by FastAPI and both global plots rendered')

            page.get_by_role('button', name='Expliquer cet individu (SHAP + LIME)', exact=True).click()
            expect(page.get_by_text('Réponse reçue de FastAPI (/explain).', exact=True)).to_be_visible()
            expect(page.get_by_text('SHAP waterfall', exact=True)).to_be_visible()
            expect(page.get_by_text('LIME (même observation)', exact=True)).to_be_visible()
            expect(page.get_by_test_id('stImage')).to_have_count(3)
            clean_page()
            capture('06_shap_lime_fastapi_navigateur.png', page.get_by_text('Probabilité estimée de panne', exact=True))
            step('Local SHAP and LIME for the same individual returned by FastAPI')

            navigate('6 · API', '6 · API d’interprétation (FastAPI)')
            expect(page.get_by_text('API connectée · Modèle disponible : True', exact=True)).to_be_visible()
            expect(page.get_by_test_id('stJson')).to_contain_text('XGBoost')
            capture('07_api_xgboost_navigateur.png', page.get_by_role('heading', name='6 · API d’interprétation (FastAPI)', exact=True))
            step('API module confirms the activated model')

            docs = context.new_page()
            docs.goto(base + '/api/docs', wait_until='domcontentloaded')
            expect(docs.locator('.swagger-ui .opblock')).to_have_count(4)
            expect(docs.get_by_role('heading', name=re.compile('ExplainMaint XAI API'))).to_be_visible()
            capture('08_swagger_public_navigateur.png', target=docs)
            docs.close()
            step('Public Swagger loads OpenAPI through /api and lists all four routes')
            if report['browser_errors']:
                raise AssertionError('The browser reported an unhandled JavaScript error; inspect browser_errors')
            report['passed'] = True
        except Exception as error:
            report['error'] = f'{type(error).__name__}: {error}'
            print(report['error'], file=sys.stderr, flush=True)
            if page is not None and not page.is_closed():
                try:
                    failure = args.output / 'failure.png'
                    page.screenshot(path=str(failure), animations='disabled')
                    artifact(failure)
                except Exception:
                    pass
        finally:
            save_report()
            if browser is not None:
                browser.close()
    print(f'Report: {report_path}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
