#!/usr/bin/env bash
# Start the local forecast app, reachable from a phone on the same Wi-Fi.
#
#   scripts/run_app.sh            # localhost only
#   scripts/run_app.sh --lan      # also reachable from other devices
#
# --lan binds 0.0.0.0, which serves the app, unauthenticated, to every device
# on the network. macOS will ask once to allow incoming connections. The
# read-only pages give nothing away; the update button is what matters, so put
# an `update_token` in .streamlit/secrets.toml before using --lan.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=".venv-app/bin/streamlit"
if [ ! -x "$PYTHON" ]; then
    echo "error: $PYTHON is missing. Create the app environment first:" >&2
    echo "  python3 -m venv .venv-app && .venv-app/bin/python -m pip install \\" >&2
    echo "      -r requirements.app.lock.txt -e ." >&2
    exit 1
fi

PORT="${PORT:-8501}"
ARGS=(run app/Home.py --server.port "$PORT")

if [ "${1:-}" = "--lan" ]; then
    # The Wi-Fi interface is not always en0 - a dock or dongle moves it - so
    # take whichever interface the default route uses.
    INTERFACE="$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')"
    ADDRESS=""
    for candidate in "$INTERFACE" en0 en1 en2; do
        [ -z "$candidate" ] && continue
        ADDRESS="$(ipconfig getifaddr "$candidate" 2>/dev/null || true)"
        [ -n "$ADDRESS" ] && break
    done
    ARGS+=(--server.address 0.0.0.0)
    echo "Serving on the local network, without authentication."
    if [ -n "$ADDRESS" ]; then
        echo "  this Mac:  http://localhost:$PORT"
        echo "  phone:     http://$ADDRESS:$PORT"
    else
        echo "  could not determine the LAN address; check System Settings > Wi-Fi"
    fi
else
    echo "Serving on http://localhost:$PORT only. Pass --lan to reach it from your phone."
fi

exec "$PYTHON" "${ARGS[@]}"
