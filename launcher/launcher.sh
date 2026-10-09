#!/bin/zsh
# Starts and stops the JARVIS server for JARVIS.app (copied into the app by build-app.sh).
#
#   launcher.sh url     the address of the interface
#   launcher.sh start   start the server and wait until it answers (exit codes below)
#   launcher.sh stop    stop the server this app started
#   launcher.sh alive   exit 0 while JARVIS answers
#
# The server is a child of the app: macOS ends it when the app quits, so the app stays
# open (in the Dock) while JARVIS runs, and quitting the app turns JARVIS off.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ROOT="${0:A:h:h:h:h}"  # .../agenteIA/JARVIS.app/Contents/Resources/launcher.sh
PORT="${JARVIS_PORT:-8300}"
URL="http://127.0.0.1:$PORT"
PY="$ROOT/backend/.venv/bin/python"
LOG="$ROOT/data/logs/jarvis.log"
PIDFILE="$ROOT/data/logs/jarvis-app.pid"

# Exit codes of "start".
NO_PROJECT=10
ALREADY_RUNNING=11  # started elsewhere (JARVIS.command): just open it
PORT_BUSY=12
NEEDS_SETUP=13      # first run or an update: JARVIS.command installs it in the Terminal
FAILED=14

is_jarvis() {
  curl -fsS -m 2 "$URL/api/health" 2>/dev/null | grep -q '"app":"jarvis"'
}

needs_setup() {
  [[ ! -x "$PY" ]] && return 0
  cmp -s "$ROOT/backend/pyproject.toml" "$ROOT/backend/.venv/.jarvis-installed" || return 0
  [[ ! -f "$ROOT/web/dist/index.html" ]] && return 0
  [[ -n "$(find "$ROOT/web/src" -newer "$ROOT/web/dist/index.html" -print -quit)" ]] && return 0
  return 1
}

our_pid() {  # the server this app started, if it is still "jarvis serve"
  local pid
  [[ -f "$PIDFILE" ]] || return 1
  pid="$(<"$PIDFILE")"
  [[ "$pid" == <-> ]] && ps -o command= -p "$pid" 2>/dev/null | grep -q "jarvis serve" && print "$pid"
}

start() {
  [[ -f "$ROOT/backend/pyproject.toml" ]] || return $NO_PROJECT
  is_jarvis && return $ALREADY_RUNNING
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && return $PORT_BUSY
  needs_setup && return $NEEDS_SETUP

  mkdir -p "${LOG:h}"
  print "\n=== $(date '+%Y-%m-%d %H:%M:%S') JARVIS.app ===" >>"$LOG"
  cd "$ROOT/backend" || return $FAILED
  PYTHONUNBUFFERED=1 "$PY" -m jarvis serve --port "$PORT" >>"$LOG" 2>&1 </dev/null &
  local server=$!
  print "$server" >"$PIDFILE"
  local i
  for i in {1..120}; do  # up to 60 s
    is_jarvis && return 0
    kill -0 "$server" 2>/dev/null || return $FAILED
    sleep 0.5
  done
  return $FAILED
}

stop() {
  local pid i
  pid="$(our_pid)" || return 0
  kill -TERM "$pid"
  for i in {1..20}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
  rm -f "$PIDFILE"
}

case "$1" in
  url) print "$URL" ;;
  start) start ;;
  stop) stop ;;
  alive) is_jarvis ;;
  *) print -u2 "usage: launcher.sh url|start|stop|alive"; exit 2 ;;
esac
