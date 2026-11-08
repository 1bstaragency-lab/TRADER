#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04/24.04 server (Hetzner, DigitalOcean, Vultr...).
# Run as your normal (non-root) user:   bash deploy/setup_ubuntu.sh
# Installs Python, the repo's packages and the background services. Places no orders.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

echo "==> System packages (needs sudo)"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git
sudo timedatectl set-timezone America/New_York   # schedules below are in New York time

echo "==> Python environment"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "!! Created .env - edit it now:  nano $REPO_DIR/.env"
fi

echo "==> Background services (run as you, keep running after you log out)"
mkdir -p ~/.config/systemd/user
for f in deploy/systemd/*; do
  sed "s#__REPO__#$REPO_DIR#g" "$f" > ~/.config/systemd/user/"$(basename "$f")"
done
sudo loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now prop-lab-backup.timer
systemctl --user enable prop-lab-dealer.timer prop-lab-solana.service

cat <<MSG

Done. Next:
  1. nano $REPO_DIR/.env                         (add your keys)
  2. .venv/bin/python scripts/check_setup.py     (everything should be PASS or SKIP)
  3. systemctl --user start prop-lab-dealer.timer     # futures paper trader, weekdays 9:00 ET
     systemctl --user start prop-lab-solana.service   # memecoin paper trader, 24/7
  4. Watch:  journalctl --user -u prop-lab-solana -f
MSG
