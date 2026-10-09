#!/bin/zsh
# JARVIS — double-click in Finder to start. Everything stays on this Mac (127.0.0.1).
# First run: creates the Python environment and builds the interface (a few minutes).
set -e
# Finder does not load the Terminal's PATH: add Homebrew and the usual places.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")"
ROOT="$PWD"
PORT="${JARVIS_PORT:-8300}"
PY="$ROOT/backend/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Primeira vez: criando o ambiente Python (backend/.venv)…"
  python3 -m venv "$ROOT/backend/.venv"
  "$PY" -m pip install --quiet --upgrade pip
fi
# Reinstall when pyproject.toml changes (marker = copy of the last installed file).
MARKER="$ROOT/backend/.venv/.jarvis-installed"
if [[ ! -f "$MARKER" ]] || ! cmp -s "$ROOT/backend/pyproject.toml" "$MARKER"; then
  echo "Instalando dependências (servidor, voz e IA local)…"
  "$PY" -m pip install --quiet -e "$ROOT/backend[server,voice,local]"
  cp "$ROOT/backend/pyproject.toml" "$MARKER"
fi

if [[ ! -f "$ROOT/web/dist/index.html" ]] || [[ -n "$(find "$ROOT/web/src" -newer "$ROOT/web/dist/index.html" -print -quit)" ]]; then
  if command -v npm >/dev/null; then
    echo "Compilando a interface…"
    (cd "$ROOT/web" && npm install --silent --no-audit --no-fund && npm run build --silent)
  else
    echo "⚠ npm não encontrado: instale o Node.js para compilar a interface."
  fi
fi

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "A porta $PORT já está em uso (o JARVIS já está aberto?). Abrindo o navegador…"
  [[ -z "$JARVIS_NO_BROWSER" ]] && open "http://127.0.0.1:$PORT"
  exit 0
fi

cd "$ROOT/backend"
if [[ -n "$JARVIS_NO_BROWSER" ]]; then
  exec "$PY" -m jarvis serve --port "$PORT"
fi
exec "$PY" -m jarvis serve --port "$PORT" --open
