#!/usr/bin/env bash
set -euo pipefail
cd /app

# Keep all services in one container; fail the container if any service exits.
pids=()
cleanup() {
    trap - EXIT INT TERM
    if ((${#pids[@]})); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

nginx -t -c /app/deployment/nginx.conf
uvicorn api:app --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips=127.0.0.1 &
pids+=("$!")
browser_host="${STREAMLIT_BROWSER_SERVER_ADDRESS:-${SPACE_HOST:-localhost}}"
browser_port="${STREAMLIT_BROWSER_SERVER_PORT:-7860}"
if [[ -n "${SPACE_HOST:-}" && -z "${STREAMLIT_BROWSER_SERVER_PORT:-}" ]]; then
    browser_port=443
fi
streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true \
    --browser.serverAddress="$browser_host" --browser.serverPort="$browser_port" \
    --browser.gatherUsageStats=false &
pids+=("$!")
nginx -c /app/deployment/nginx.conf -g 'daemon off;' &
pids+=("$!")

status=0
wait -n "${pids[@]}" || status=$?
echo "A service stopped (status ${status}); stopping the container." >&2
# Even a clean but unexpected service exit must restart the complete stack.
if ((status == 0)); then status=1; fi
exit "$status"
