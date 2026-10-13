"""Check a local Docker URL or a deployed Space without changing its model.

Uses only the Python standard library. Run after activating a model with
--require-model to also verify predictions, SHAP, LIME and global explanations.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import socket
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_RECORD = {
    'Type': 'M',
    'Air temperature [K]': 300.0,
    'Process temperature [K]': 310.0,
    'Rotational speed [rpm]': 1500,
    'Torque [Nm]': 40.0,
    'Tool wear [min]': 150,
}


def fetch(base, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    headers = {'User-Agent': 'ExplainMaint-deployment-check/1.0'}
    if data is not None:
        headers['Content-Type'] = 'application/json'
    request = Request(base + path, data=data, headers=headers)
    try:
        response = urlopen(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read()
        content_type = response.headers.get('Content-Type', '')
        body = json.loads(raw) if 'application/json' in content_type else raw.decode('utf-8', errors='replace')
        return response.status, body


def check_websocket(base, timeout=20):
    """Validate the HTTP upgrade and accept key, then close the connection."""
    url = urlsplit(base)
    key = base64.b64encode(os.urandom(16)).decode('ascii')
    port = url.port or (443 if url.scheme == 'https' else 80)
    path = url.path.rstrip('/') + '/_stcore/stream'
    request = (
        f'GET {path} HTTP/1.1\r\nHost: {url.netloc}\r\n'
        f'Origin: {url.scheme}://{url.netloc}\r\n'
        'Upgrade: websocket\r\nConnection: Upgrade\r\n'
        f'Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n'
        'Sec-WebSocket-Protocol: streamlit\r\n\r\n'
    )
    connection = socket.create_connection((url.hostname, port), timeout=timeout)
    if url.scheme == 'https':
        connection = ssl.create_default_context().wrap_socket(connection, server_hostname=url.hostname)
    with connection:
        connection.sendall(request.encode('ascii'))
        raw = b''
        while b'\r\n\r\n' not in raw:
            chunk = connection.recv(4096)
            if not chunk or len(raw) > 65536:
                raise RuntimeError('Incomplete WebSocket handshake')
            raw += chunk
        lines = raw.split(b'\r\n\r\n', 1)[0].decode('latin-1').split('\r\n')
        if lines[0].split()[1] != '101':
            raise RuntimeError('WebSocket upgrade failed: ' + lines[0])
        headers = dict(line.split(':', 1) for line in lines[1:] if ':' in line)
        headers = {name.lower(): value.strip() for name, value in headers.items()}
        expected = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
        if headers.get('sec-websocket-accept') != expected:
            raise RuntimeError('Invalid WebSocket accept key')
        # Empty, masked client close frame (RFC 6455).
        connection.sendall(b'\x88\x80' + os.urandom(4))


def verify(base, require_model=False, record=None):
    checks = []
    model_available = False

    def check(name, operation):
        try:
            operation()
            checks.append({'name': name, 'status': 'passed'})
        except Exception as error:
            checks.append({'name': name, 'status': 'failed', 'detail': str(error)})

    def expect(condition, message):
        if not condition:
            raise AssertionError(message)

    def interface():
        status, body = fetch(base, '/')
        expect(status == 200 and 'streamlit' in body.lower(), 'Streamlit HTML was not served')

    def streamlit_health():
        status, body = fetch(base, '/_stcore/health')
        expect(status == 200 and body.strip() == 'ok', 'Streamlit health check failed')

    def api_health():
        nonlocal model_available
        status, body = fetch(base, '/api/health')
        expect(status == 200 and body.get('status') == 'ok', 'FastAPI health check failed')
        model_available = body.get('trained_model_available') is True
        if require_model:
            expect(model_available, 'Activate a model in Streamlit before the full check')

    def openapi():
        status, body = fetch(base, '/api/openapi.json')
        expect(status == 200, 'OpenAPI schema is inaccessible')
        expect({'/health', '/model', '/global', '/explain'} <= set(body['paths']), 'Missing API routes')
        expect(any(server['url'] == '/api' for server in body.get('servers', [])), 'OpenAPI root_path must be /api')
        status, docs = fetch(base, '/api/docs')
        expect(status == 200 and '/api/openapi.json' in docs, 'Swagger must load its schema through /api')

    def model():
        status, body = fetch(base, '/api/model')
        if model_available:
            expect(status == 200 and body.get('features') and body.get('name'), 'Model metadata is invalid')
        else:
            expect(status == 404, 'An inactive model must return HTTP 404')

    def explanations():
        status, body = fetch(base, '/api/explain', {'record': record or DEFAULT_RECORD, 'threshold': .5}, timeout=120)
        expect(status == 200, 'Explanation request failed; use --record-json for a custom feature set')
        probability = body['probability']
        expect(math.isfinite(probability) and 0 <= probability <= 1, 'Invalid probability')
        expect(len(body['names']) == len(body['shap_values']) > 0, 'Invalid SHAP dimensions')
        expect(math.isclose(body['base_value'] + sum(body['shap_values']), probability, abs_tol=1e-4), 'SHAP additivity failed')
        expect(body['lime'] and all(math.isfinite(item['weight']) for item in body['lime']), 'Missing or invalid LIME explanation')
        expect(body['decision'] == int(probability >= .5), 'Threshold decision is inconsistent')
        status, global_body = fetch(base, '/api/global?n=2', timeout=120)
        expect(status == 200 and len(global_body['values']) > 0, 'Global SHAP is unavailable')
        expect(all(len(row) == len(global_body['names']) for row in global_body['values']), 'Invalid global SHAP dimensions')

    check('streamlit_html', interface)
    check('streamlit_health', streamlit_health)
    check('streamlit_websocket', lambda: check_websocket(base))
    check('fastapi_health', api_health)
    check('openapi_and_swagger', openapi)
    check('model_metadata', model)
    if model_available:
        check('prediction_shap_lime_global', explanations)
    else:
        checks.append({'name': 'prediction_shap_lime_global', 'status': 'skipped',
                       'detail': 'No active model; train and activate it in Streamlit, then rerun with --require-model'})
    return {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_url': base,
        'model_available': model_available,
        'scope': 'full' if model_available else 'services_only',
        'passed': all(item['status'] != 'failed' for item in checks),
        'checks': checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base_url', help='Public app URL ending in .hf.space, or http://localhost:7860')
    parser.add_argument('--require-model', action='store_true')
    parser.add_argument('--record-json', type=Path, help='JSON object containing the active model feature values')
    parser.add_argument('--output', type=Path, help='Save the actual checks as JSON')
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    url = urlsplit(base)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        parser.error('Use an HTTP(S) URL without credentials, query or fragment')
    record = json.loads(args.record_json.read_text(encoding='utf-8')) if args.record_json else None
    report = verify(base, args.require_model, record)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
