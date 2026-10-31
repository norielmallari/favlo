#!/bin/sh
set -eu
pid=""
cleanup() {
  if [ -n "$pid" ]; then
    kill -INT "$pid" 2>/dev/null || true
    i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 40 ]; do
      sleep 0.25
      i=$((i + 1))
    done
    kill -TERM "$pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
uv run favlo web &
pid=$!
wait "$pid"
