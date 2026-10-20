#!/usr/bin/env bash
# Offline, no installation, <=10 minutes, normal priority. Apple-silicon Mac only.
set -eu
: "${GLIMMER_PYTHON:?Set GLIMMER_PYTHON to the absolute Python path of an existing MLX-LM venv}"
: "${GLIMMER_MODEL:?Set GLIMMER_MODEL to an existing local MLX 4-bit Glimmer directory}"
case "$GLIMMER_PYTHON" in /*) ;; *) echo "GLIMMER_PYTHON must be absolute" >&2; exit 2;; esac
[ -x "$GLIMMER_PYTHON" ] || { echo "Python interpreter is not installed at that path" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
exec "$GLIMMER_PYTHON" "$HERE/infer_mac_glimmer.py"   --model "$GLIMMER_MODEL"   --out "${GLIMMER_OUT:-$PWD/infer-mac-results}"
