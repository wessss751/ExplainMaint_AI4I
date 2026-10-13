"""Docker liveness check for both services through the public reverse proxy."""
import json
from urllib.request import urlopen


def main():
    with urlopen('http://127.0.0.1:7860/_stcore/health', timeout=5) as response:
        if response.status != 200 or response.read().strip() != b'ok':
            raise RuntimeError('Streamlit is not healthy')
    with urlopen('http://127.0.0.1:7860/api/health', timeout=5) as response:
        if response.status != 200 or json.load(response).get('status') != 'ok':
            raise RuntimeError('FastAPI is not healthy')


if __name__ == '__main__':
    main()
