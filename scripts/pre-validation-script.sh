#!/usr/bin/env bash
#
# pre-validation-script.sh — Extended OpenEnv Submission Validator
#
# This script starts from the official hackathon template and adds repo-level
# checks for required files, inference env-var contract, structured stdout log
# markers, Dockerfile presence, and OpenEnv submission shape.
#
# Usage:
#   ./scripts/pre-validation-script.sh <ping_url> [repo_dir]
#
# Examples:
#   ./scripts/pre-validation-script.sh https://my-team.hf.space
#   ./scripts/pre-validation-script.sh https://my-team.hf.space /path/to/repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
VALIDATOR_IMAGE_TAG="openenv-submission-validator:local"
PY311_HELPER_IMAGE="python:3.11-slim"
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$secs" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi

PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
warn() { log "${YELLOW}WARN${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  if [ -f "$path" ]; then
    pass "$label present: ${path#$REPO_DIR/}"
  else
    fail "$label missing: ${path#$REPO_DIR/}"
    return 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [ -d "$path" ]; then
    pass "$label present: ${path#$REPO_DIR/}"
  else
    fail "$label missing: ${path#$REPO_DIR/}"
    return 1
  fi
}

require_grep() {
  local pattern="$1"
  local file="$2"
  local label="$3"
  if grep -Eq "$pattern" "$file"; then
    pass "$label"
  else
    fail "$label"
    return 1
  fi
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

log "${BOLD}Step 1/8: Repo structure checks${NC} ..."

require_file "$REPO_DIR/README.md" "README" || stop_at "Step 1"
require_file "$REPO_DIR/inference.py" "Root inference script" || stop_at "Step 1"
require_file "$REPO_DIR/openenv.yaml" "openenv.yaml" || stop_at "Step 1"
require_file "$REPO_DIR/Dockerfile" "Dockerfile" || stop_at "Step 1"
require_file "$REPO_DIR/models.py" "Root models.py" || stop_at "Step 1"
require_dir "$REPO_DIR/server" "server package" || stop_at "Step 1"
require_dir "$REPO_DIR/tests" "tests directory" || stop_at "Step 1"

log "${BOLD}Step 2/8: Inference contract checks${NC} ..."

require_grep 'from openai import OpenAI' "$REPO_DIR/inference.py" "OpenAI client imported in inference.py" || stop_at "Step 2"
require_grep 'API_BASE_URL *= *os.getenv\("API_BASE_URL",' "$REPO_DIR/inference.py" "API_BASE_URL read from env with default" || stop_at "Step 2"
require_grep 'MODEL_NAME *= *os.getenv\("MODEL_NAME",' "$REPO_DIR/inference.py" "MODEL_NAME read from env with default" || stop_at "Step 2"
require_grep 'HF_TOKEN *= *os.getenv\("HF_TOKEN"\)' "$REPO_DIR/inference.py" "HF_TOKEN read from env without default" || stop_at "Step 2"
require_grep 'LOCAL_IMAGE_NAME *= *os.getenv\("LOCAL_IMAGE_NAME"\)' "$REPO_DIR/inference.py" "LOCAL_IMAGE_NAME optionally supported" || stop_at "Step 2"
require_grep 'OpenAI\(base_url=API_BASE_URL, api_key=HF_TOKEN\)' "$REPO_DIR/inference.py" "OpenAI client configured from required env vars" || stop_at "Step 2"
require_grep '\[START\]' "$REPO_DIR/inference.py" "Structured START log marker present" || stop_at "Step 2"
require_grep '\[STEP\]' "$REPO_DIR/inference.py" "Structured STEP log marker present" || stop_at "Step 2"
require_grep '\[END\]' "$REPO_DIR/inference.py" "Structured END log marker present" || stop_at "Step 2"

log "${BOLD}Step 3/8: OpenEnv spec surface checks${NC} ..."

require_grep '^spec_version:' "$REPO_DIR/openenv.yaml" "openenv.yaml declares spec_version" || stop_at "Step 3"
require_grep '^runtime:' "$REPO_DIR/openenv.yaml" "openenv.yaml declares runtime" || stop_at "Step 3"
require_grep '^app:' "$REPO_DIR/openenv.yaml" "openenv.yaml declares app entrypoint" || stop_at "Step 3"
require_grep '^port:' "$REPO_DIR/openenv.yaml" "openenv.yaml declares port" || stop_at "Step 3"
require_grep 'def reset\(' "$REPO_DIR/server/scheme_env_environment.py" "Environment defines reset()" || stop_at "Step 3"
require_grep 'def step\(' "$REPO_DIR/server/scheme_env_environment.py" "Environment defines step()" || stop_at "Step 3"
require_grep 'def state\(' "$REPO_DIR/server/scheme_env_environment.py" "Environment exposes state property/method" || stop_at "Step 3"

TASK_COUNT=$(grep -Ec 'task_id == [1-9]|Task [1-9]|TASK [1-9]/' "$REPO_DIR/server/scheme_env_environment.py" || true)
if [ "$TASK_COUNT" -ge 3 ]; then
  pass "Detected 3+ task definitions in environment logic"
else
  fail "Could not detect 3+ task definitions in environment logic"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/8: README submission-content checks${NC} ..."

require_grep 'Action Space' "$REPO_DIR/README.md" "README documents action space" || stop_at "Step 4"
require_grep 'Observation Space' "$REPO_DIR/README.md" "README documents observation space" || stop_at "Step 4"
require_grep 'Setup and Running|Setup' "$REPO_DIR/README.md" "README documents setup instructions" || stop_at "Step 4"
require_grep 'The 5 Tasks|Tasks' "$REPO_DIR/README.md" "README documents tasks" || stop_at "Step 4"
require_grep 'Baseline Results|leaderboard.csv|results.json' "$REPO_DIR/README.md" "README documents baseline outputs" || stop_at "Step 4"

log "${BOLD}Step 5/8: Pinging HF Space${NC} ($PING_URL/reset) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 5"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 5"
fi

HEALTH_OUTPUT=$(portable_mktemp "validate-health")
CLEANUP_FILES+=("$HEALTH_OUTPUT")
HEALTH_CODE=$(curl -s -o "$HEALTH_OUTPUT" -w "%{http_code}" \
  "$PING_URL/health" --max-time 15 || printf "000")
if [ "$HEALTH_CODE" = "200" ]; then
  pass "HF Space /health responds with HTTP 200"
else
  fail "HF Space /health returned HTTP $HEALTH_CODE"
  stop_at "Step 5"
fi

log "${BOLD}Step 6/8: Running docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 6"
fi

BUILD_LOG=$(portable_mktemp "validate-docker-build")
CLEANUP_FILES+=("$BUILD_LOG")
BUILD_OK=false
if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t "$VALIDATOR_IMAGE_TAG" "$REPO_DIR" >"$BUILD_LOG" 2>&1; then
  BUILD_OK=true
fi

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  tail -20 "$BUILD_LOG"
  stop_at "Step 6"
fi

log "${BOLD}Step 7/8: Running openenv validate${NC} ..."

OPENENV_VALIDATE_CMD=()
if command -v openenv >/dev/null 2>&1; then
  OPENENV_VALIDATE_CMD=(openenv validate)
elif command -v python >/dev/null 2>&1 && python -c "import openenv" >/dev/null 2>&1; then
  OPENENV_VALIDATE_CMD=(python -m openenv validate)
elif command -v python >/dev/null 2>&1 && python -c "import openenv_core" >/dev/null 2>&1; then
  OPENENV_VALIDATE_CMD=(python -m openenv_core validate)
elif command -v docker >/dev/null 2>&1; then
  OPENENV_VALIDATE_CMD=(
    docker run --rm
    -v "$REPO_DIR:/workspace"
    "$PY311_HELPER_IMAGE"
    sh -lc
    "pip install 'openenv-core[cli]>=0.2.3' >/tmp/openenv-install.log 2>&1 && cd /workspace && openenv validate"
  )
else
  fail "openenv command/module not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 7"
fi

VALIDATE_LOG=$(portable_mktemp "validate-openenv")
CLEANUP_FILES+=("$VALIDATE_LOG")
VALIDATE_OK=false
if (cd "$REPO_DIR" && "${OPENENV_VALIDATE_CMD[@]}" >"$VALIDATE_LOG" 2>&1); then
  VALIDATE_OK=true
fi

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  if [ -s "$VALIDATE_LOG" ]; then
    while IFS= read -r line; do
      log "  $line"
    done < "$VALIDATE_LOG"
  fi
else
  fail "openenv validate failed"
  cat "$VALIDATE_LOG"
  stop_at "Step 7"
fi

log "${BOLD}Step 8/8: Local quality checks${NC} ..."

PYTHON_BIN=""
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

HOST_PYTHON_OK=false
if [ -n "$PYTHON_BIN" ] && "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
  HOST_PYTHON_OK=true
fi

if [ "$HOST_PYTHON_OK" = true ]; then
  PY_COMPILE_OK=true
  PY_COMPILE_LOG=$(portable_mktemp "validate-pycompile")
  CLEANUP_FILES+=("$PY_COMPILE_LOG")
  (cd "$REPO_DIR" && "$PYTHON_BIN" -m py_compile inference.py models.py server/*.py >"$PY_COMPILE_LOG" 2>&1) || PY_COMPILE_OK=false
  if [ "$PY_COMPILE_OK" = true ]; then
    pass "Key Python files compile cleanly"
  else
    fail "Python compile check failed"
    cat "$PY_COMPILE_LOG"
    stop_at "Step 8"
  fi
elif command -v docker >/dev/null 2>&1; then
  PY_COMPILE_OK=true
  PY_COMPILE_LOG=$(portable_mktemp "validate-pycompile")
  CLEANUP_FILES+=("$PY_COMPILE_LOG")
  docker run --rm -v "$REPO_DIR:/workspace" "$PY311_HELPER_IMAGE" \
    sh -lc "cd /workspace && python -m py_compile inference.py models.py server/*.py" \
    >"$PY_COMPILE_LOG" 2>&1 || PY_COMPILE_OK=false
  if [ "$PY_COMPILE_OK" = true ]; then
    pass "Key Python files compile cleanly"
  else
    fail "Python compile check failed"
    cat "$PY_COMPILE_LOG"
    stop_at "Step 8"
  fi
else
  warn "python not found; skipping py_compile check"
fi

if command -v pytest >/dev/null 2>&1; then
  TEST_OK=true
  TEST_LOG=$(portable_mktemp "validate-pytest")
  CLEANUP_FILES+=("$TEST_LOG")
  (cd "$REPO_DIR" && pytest tests/ -q >"$TEST_LOG" 2>&1) || TEST_OK=false
  if [ "$TEST_OK" = true ]; then
    pass "pytest tests/ passed"
  else
    fail "pytest tests/ failed"
    cat "$TEST_LOG"
    stop_at "Step 8"
  fi
elif command -v docker >/dev/null 2>&1; then
  TEST_OK=true
  TEST_LOG=$(portable_mktemp "validate-pytest")
  CLEANUP_FILES+=("$TEST_LOG")
  docker run --rm -v "$REPO_DIR:/workspace" "$PY311_HELPER_IMAGE" \
    sh -lc "pip install -r /workspace/requirements.txt pytest >/tmp/pytest-install.log 2>&1 && cd /workspace && PYTHONPATH=. pytest tests/ -q" \
    >"$TEST_LOG" 2>&1 || TEST_OK=false
  if [ "$TEST_OK" = true ]; then
    pass "pytest tests/ passed"
  else
    fail "pytest tests/ failed"
    cat "$TEST_LOG"
    stop_at "Step 8"
  fi
else
  warn "pytest not installed; skipping test execution"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  Validation checks passed: %d${NC}\n" "$PASS"
printf "${GREEN}${BOLD}  Submission looks ready for hackathon review.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
