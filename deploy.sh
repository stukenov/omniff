#!/usr/bin/env bash
set -euo pipefail

SERVER="root@164.138.46.36"
PORT=15126
REMOTE_DIR="/root/omniff"

echo "=== Syncing to KazNU ==="
rsync -avz --exclude='target/' --exclude='.git/' --exclude='__pycache__/' \
  -e "ssh -p $PORT -o ConnectTimeout=120 -o ServerAliveInterval=15" \
  . "$SERVER:$REMOTE_DIR/"

echo "=== Installing on server ==="
ssh -p "$PORT" -o ConnectTimeout=120 -o ServerAliveInterval=15 "$SERVER" \
  "cd $REMOTE_DIR/python && pip install -e '.[all]' 2>&1 | tail -5"

echo "=== Running unit tests ==="
ssh -p "$PORT" -o ConnectTimeout=120 -o ServerAliveInterval=15 "$SERVER" \
  "cd $REMOTE_DIR && python3 -m pytest tests/python/unit/ -v 2>&1"

echo "=== Done ==="
