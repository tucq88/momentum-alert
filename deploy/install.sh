#!/usr/bin/env bash
# Install the scanner as a systemd timer on a fresh VM (Debian/Ubuntu).
#
#   sudo ./deploy/install.sh
#
# Idempotent: safe to re-run after a git pull to pick up new code.
set -euo pipefail

APP_DIR=/opt/rsi-alert
STATE_DIR=/var/lib/rsi-alert
ENV_FILE=/etc/rsi-alert.env
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }

echo "==> service user"
id -u rsi >/dev/null 2>&1 || useradd --system --home "$STATE_DIR" --shell /usr/sbin/nologin rsi

echo "==> uv"
# uv is the only runtime dependency: the scanner declares its own Python
# packages inline (PEP 723), so there is no virtualenv to build or maintain.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

echo "==> code -> $APP_DIR"
install -d -o rsi -g rsi "$APP_DIR" "$STATE_DIR" "$STATE_DIR/site" "$STATE_DIR/csv"
install -o rsi -g rsi -m 0755 "$SRC/rsi_scanner.py" "$APP_DIR/"
install -o rsi -g rsi -m 0644 "$SRC/dashboard.html" "$SRC/dashboard.js" "$APP_DIR/"

echo "==> config"
if [ ! -f "$ENV_FILE" ]; then
  install -m 0600 "$SRC/deploy/rsi-alert.env.example" "$ENV_FILE"
  echo "    wrote $ENV_FILE -- EDIT IT before the first run (it has placeholder secrets)"
fi

echo "==> warming the uv cache as the service user"
# Do the first dependency download now, interactively, rather than inside a
# timer run where a failure is just a red unit in the journal.
sudo -u rsi env HOME="$STATE_DIR" "$APP_DIR/rsi_scanner.py" --help >/dev/null

echo "==> systemd"
install -m 0644 "$SRC/deploy/rsi-alert.service" "$SRC/deploy/rsi-alert.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now rsi-alert.timer

cat <<'DONE'

Installed. What to do next:

  1. Edit /etc/rsi-alert.env   (bot token, chat id, and the RSI_FLAGS line)
  2. Run it once by hand:      sudo systemctl start rsi-alert.service
  3. Watch it:                 journalctl -u rsi-alert.service -f
  4. Confirm the schedule:     systemctl list-timers rsi-alert.timer

The first run is the slow one: a cold candle cache fetches full windows for
every symbol, and it records every crossing it finds WITHOUT alerting on the old
ones, so you get a seeded history instead of 300 notifications.

Outcomes:  sudo -u rsi /opt/rsi-alert/rsi_scanner.py --offline --report --no-csv
DONE
