#!/usr/bin/env bash
# Start Gemma 4 API server
# Usage: ./start.sh [--port PORT] [--host HOST]

cd "$(dirname "$0")"
source .venv/bin/activate
exec python server.py "$@"
