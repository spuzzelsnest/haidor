#!/usr/bin/env bash
# Installs the pi-agent web server as a systemd service ("haidar").
# Run from the folder that contains server.py and agent.py:
#     sudo ./install-haidar.sh
set -euo pipefail

SERVICE=haidar
LOG=/var/log/haidar.log

[ "$EUID" -eq 0 ] || { echo "Please run with sudo."; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"

[ -f "$APP_DIR/server.py" ] || { echo "server.py not found in $APP_DIR"; exit 1; }

echo "App dir : $APP_DIR"
echo "Run as  : $RUN_USER"
echo "Log file: $LOG"

# --- log file ---------------------------------------------------------------
touch "$LOG"
chmod 644 "$LOG"

# --- systemd unit -----------------------------------------------------------
cat > "/etc/systemd/system/${SERVICE}.service" <<EOF
[Unit]
Description=Haidar - pi-agent web server
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 -u ${APP_DIR}/server.py
Restart=on-failure
RestartSec=5
StandardOutput=append:${LOG}
StandardError=append:${LOG}

[Install]
WantedBy=multi-user.target
EOF

# --- log rotation (copytruncate: systemd keeps the file open) ---------------
cat > "/etc/logrotate.d/${SERVICE}" <<EOF
${LOG} {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# --- enable & start ---------------------------------------------------------
systemctl daemon-reload
systemctl enable --now "${SERVICE}.service"
sleep 1
systemctl --no-pager status "${SERVICE}.service" || true

echo
echo "Done. Follow the log with:  tail -f ${LOG}"
