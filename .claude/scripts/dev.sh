#!/usr/bin/env bash
# dev.sh — Manage development background services for Attractor
# Usage: bash .claude/scripts/dev.sh [start|stop|status] [tests|lint|all]
#
# Can be run by Claude via /project:dev or directly by developers.

set -euo pipefail

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

PID_TEST="$SCRIPT_DIR/test-watcher.pid"
PID_LINT="$SCRIPT_DIR/lint-watcher.pid"
LOG_TEST="$SCRIPT_DIR/test-watcher.log"
LOG_LINT="$SCRIPT_DIR/lint-watcher.log"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Helpers ---
is_running() {
    local pidfile="$1"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            # Stale PID file
            rm -f "$pidfile"
            return 1
        fi
    fi
    return 1
}

ensure_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        error "No virtual environment found at $VENV_DIR"
        error "Run: bash .claude/scripts/install.sh  (or /project:install)"
        exit 1
    fi
}

# --- Commands ---

cmd_start_tests() {
    ensure_venv
    if is_running "$PID_TEST"; then
        warn "Test watcher already running (PID $(cat "$PID_TEST"))"
        return
    fi

    # Install pytest-watcher on demand
    if ! "$VENV_DIR/bin/python" -c "import pytest_watcher" 2>/dev/null; then
        info "Installing pytest-watcher..."
        if command -v uv &>/dev/null; then
            uv pip install pytest-watcher
        else
            "$VENV_DIR/bin/pip" install pytest-watcher
        fi
    fi

    info "Starting test watcher..."
    cd "$PROJECT_ROOT"
    nohup "$VENV_DIR/bin/ptw" "$PROJECT_ROOT" --now --runner "$VENV_DIR/bin/pytest" -- -q --tb=short > "$LOG_TEST" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_TEST"
    ok "Test watcher started (PID $pid) — log: $LOG_TEST"
}

cmd_start_lint() {
    ensure_venv
    if is_running "$PID_LINT"; then
        warn "Lint watcher already running (PID $(cat "$PID_LINT"))"
        return
    fi

    info "Starting lint watcher (ruff + black every 10s)..."
    cd "$PROJECT_ROOT"
    nohup bash -c "
        while true; do
            \"$VENV_DIR/bin/ruff\" check --fix \"$PROJECT_ROOT\" 2>&1 || true
            \"$VENV_DIR/bin/black\" --quiet \"$PROJECT_ROOT\" 2>&1 || true
            sleep 10
        done
    " > "$LOG_LINT" 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_LINT"
    ok "Lint watcher started (PID $pid) — log: $LOG_LINT"
}

cmd_stop_tests() {
    if is_running "$PID_TEST"; then
        local pid
        pid=$(cat "$PID_TEST")
        info "Stopping test watcher (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        # Also kill child processes
        pkill -P "$pid" 2>/dev/null || true
        rm -f "$PID_TEST"
        ok "Test watcher stopped"
    else
        info "Test watcher is not running"
    fi
}

cmd_stop_lint() {
    if is_running "$PID_LINT"; then
        local pid
        pid=$(cat "$PID_LINT")
        info "Stopping lint watcher (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        pkill -P "$pid" 2>/dev/null || true
        rm -f "$PID_LINT"
        ok "Lint watcher stopped"
    else
        info "Lint watcher is not running"
    fi
}

cmd_status() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  Attractor Dev Environment Status${NC}"
    echo -e "${CYAN}========================================${NC}"

    # Python / venv
    if [[ -d "$VENV_DIR" ]]; then
        local pyver
        pyver=$("$VENV_DIR/bin/python" --version 2>&1 || echo "unknown")
        echo -e "  Python:       ${GREEN}$pyver${NC}"
        echo -e "  Venv:         $VENV_DIR"
    else
        echo -e "  Python:       ${RED}No venv found${NC}"
        echo -e "  Venv:         ${RED}$VENV_DIR (missing)${NC}"
    fi

    # Installer
    if command -v uv &>/dev/null; then
        echo -e "  Installer:    ${GREEN}uv${NC}"
    else
        echo -e "  Installer:    pip"
    fi

    # API keys (check .env, never print values)
    echo ""
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        for key in ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY; do
            val=$(grep "^${key}=" "$PROJECT_ROOT/.env" 2>/dev/null | cut -d'=' -f2- || true)
            if [[ -n "$val" ]]; then
                echo -e "  $key: ${GREEN}set${NC}"
            else
                echo -e "  $key: ${YELLOW}not set${NC}"
            fi
        done
    else
        echo -e "  .env:         ${YELLOW}not found${NC}"
    fi

    # Running services
    echo ""
    if is_running "$PID_TEST"; then
        echo -e "  Test watcher: ${GREEN}running${NC} (PID $(cat "$PID_TEST"))"
    else
        echo -e "  Test watcher: ${YELLOW}stopped${NC}"
    fi

    if is_running "$PID_LINT"; then
        echo -e "  Lint watcher: ${GREEN}running${NC} (PID $(cat "$PID_LINT"))"
    else
        echo -e "  Lint watcher: ${YELLOW}stopped${NC}"
    fi

    # Test count
    echo ""
    if [[ -d "$VENV_DIR" ]]; then
        local test_count
        test_count=$("$VENV_DIR/bin/python" -m pytest --collect-only -q 2>/dev/null | tail -1 || echo "unknown")
        echo -e "  Tests:        $test_count"
    fi

    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# --- Main ---
ACTION="${1:-status}"
TARGET="${2:-all}"

case "$ACTION" in
    start)
        case "$TARGET" in
            tests) cmd_start_tests ;;
            lint)  cmd_start_lint ;;
            all)   cmd_start_tests; cmd_start_lint ;;
            *)     error "Unknown target: $TARGET (use tests, lint, or all)"; exit 1 ;;
        esac
        ;;
    stop)
        case "$TARGET" in
            tests) cmd_stop_tests ;;
            lint)  cmd_stop_lint ;;
            all)   cmd_stop_tests; cmd_stop_lint ;;
            *)     error "Unknown target: $TARGET (use tests, lint, or all)"; exit 1 ;;
        esac
        ;;
    status)
        cmd_status
        ;;
    *)
        error "Unknown action: $ACTION"
        echo "Usage: $0 [start|stop|status] [tests|lint|all]"
        exit 1
        ;;
esac
