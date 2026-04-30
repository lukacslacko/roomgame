#!/usr/bin/env bash
# Drive the Win11 compute box (192.168.1.213 by default) as a remote backend
# from this Mac. Lives in the repo so both sides can read what the other does.
#
#   tools/remote/remote.sh run "<cmd>"          # run cmd in repo dir on Win11
#   tools/remote/remote.sh py tools/foo.py args # run a python script via the
#                                                 repo's .venv (auto picks
#                                                 .venv/Scripts on git-bash,
#                                                 .venv/bin on wsl)
#   tools/remote/remote.sh push <path> [<path>] # rsync local paths -> Win11
#   tools/remote/remote.sh pull <path>          # rsync Win11 path -> local cwd
#   tools/remote/remote.sh sync-frames          # captured_frames/ Mac -> Win11
#   tools/remote/remote.sh pull-artifacts <sess>
#                                                 # standard derived dirs back
#   tools/remote/remote.sh shell                # interactive ssh
#   tools/remote/remote.sh check                # quick connectivity probe

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/config.sh"

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing $CONFIG — copy config.sh.example to config.sh and edit." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"

: "${WIN11_HOST:?set WIN11_HOST in config.sh}"
: "${WIN11_USER:?set WIN11_USER in config.sh}"
: "${WIN11_REPO:?set WIN11_REPO in config.sh}"
: "${WIN11_SHELL:=gitbash}"
: "${WIN11_SSH_ARGS:=}"

ssh_target="$WIN11_USER@$WIN11_HOST"
# shellcheck disable=SC2206
ssh_args=( $WIN11_SSH_ARGS )

# Wrap a command line so it executes inside the repo with the right login shell.
remote_wrap() {
  local cmd="$*"
  case "$WIN11_SHELL" in
    gitbash)
      # Default Win11 OpenSSH shell is cmd.exe; explicitly invoke git-bash so
      # forward-slash paths and POSIX tooling work.
      printf '"C:\\Program Files\\Git\\bin\\bash.exe" -lc %q' \
        "cd '$WIN11_REPO' && $cmd"
      ;;
    wsl)
      printf 'wsl.exe bash -lc %q' "cd '$WIN11_REPO' && $cmd"
      ;;
    powershell)
      printf 'powershell -NoProfile -Command %q' \
        "Set-Location -LiteralPath '$WIN11_REPO'; $cmd"
      ;;
    *) echo "Unknown WIN11_SHELL=$WIN11_SHELL" >&2; exit 1 ;;
  esac
}

cmd="${1:-}"; shift || true
case "$cmd" in
  run)
    [[ $# -ge 1 ]] || { echo "usage: run <cmd...>" >&2; exit 1; }
    # shellcheck disable=SC2046
    ssh "${ssh_args[@]}" "$ssh_target" $(remote_wrap "$@")
    ;;
  py)
    [[ $# -ge 1 ]] || { echo "usage: py <script> [args...]" >&2; exit 1; }
    case "$WIN11_SHELL" in
      gitbash)    py=".venv/Scripts/python.exe" ;;
      wsl)        py=".venv/bin/python" ;;
      powershell) py=".venv\\Scripts\\python.exe" ;;
    esac
    # shellcheck disable=SC2046
    ssh "${ssh_args[@]}" "$ssh_target" $(remote_wrap "$py $*")
    ;;
  push)
    [[ $# -ge 1 ]] || { echo "usage: push <path> [<path>...]" >&2; exit 1; }
    rsync -av --progress "$@" "$ssh_target:$WIN11_REPO/"
    ;;
  pull)
    [[ $# -ge 1 ]] || { echo "usage: pull <path>" >&2; exit 1; }
    rsync -av --progress "$ssh_target:$WIN11_REPO/$1" .
    ;;
  sync-frames)
    rsync -av --progress --delete-after \
      "$REPO/captured_frames/" \
      "$ssh_target:$WIN11_REPO/captured_frames/"
    ;;
  pull-artifacts)
    [[ $# -eq 1 ]] || { echo "usage: pull-artifacts <session_id>" >&2; exit 1; }
    sess="$1"
    for sub in model_raw_v3 model_raw_splat frames_pose_scratch \
               frames_pose_aligned features_meta.json; do
      src="$ssh_target:$WIN11_REPO/captured_frames/$sess/$sub"
      dst="$REPO/captured_frames/$sess/"
      echo "=== $sub ==="
      rsync -av --progress "$src" "$dst" || echo "  (skipped — not present)"
    done
    ;;
  shell)
    ssh "${ssh_args[@]}" -t "$ssh_target" \
      $(remote_wrap "exec \$SHELL -l")
    ;;
  check)
    echo "[$ssh_target] uname / pwd / python / nvidia-smi"
    # shellcheck disable=SC2046
    ssh "${ssh_args[@]}" "$ssh_target" $(remote_wrap '
      uname -a
      pwd
      git rev-parse --abbrev-ref HEAD 2>/dev/null || true
      command -v python && python --version || true
      command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(no nvidia-smi)"
    ')
    ;;
  ""|-h|--help)
    sed -n '2,16p' "$0"
    ;;
  *)
    echo "Unknown subcommand: $cmd" >&2
    sed -n '2,16p' "$0" >&2
    exit 1
    ;;
esac
