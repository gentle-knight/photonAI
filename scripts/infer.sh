#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 1 ]]; then
  echo "用法: $0 <input.csv|txt> [--run-dir path] [--output path]" >&2
  exit 1
fi
python -m src.main infer --config configs/default.yaml --input "$1" "${@:2}"
