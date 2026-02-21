#!/usr/bin/env bash
# install.sh — Set up the Attractor development environment
# Usage: bash .claude/scripts/install.sh [--fresh|fresh]
#
# Can be run by Claude via /project:install or directly by developers.

set -euo pipefail

# --- Configuration ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
MIN_PYTHON_VERSION="3.11"
ENV_FILE="$PROJECT_ROOT/.env"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# --- Parse arguments ---
FRESH=false
for arg in "$@"; do
    case "$arg" in
        --fresh|fresh) FRESH=true ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

cd "$PROJECT_ROOT"

# --- Step 1: Find Python >= 3.11 ---
info "Searching for Python >= $MIN_PYTHON_VERSION..."
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        if [[ -n "$version" ]]; then
            major="${version%%.*}"
            minor="${version#*.}"
            if (( major == 3 && minor >= 11 )); then
                PYTHON="$candidate"
                ok "Found $candidate ($version)"
                break
            fi
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    error "Python >= $MIN_PYTHON_VERSION not found."
    error "Install it with: brew install python@3.13  or  pyenv install 3.13"
    exit 1
fi

# --- Step 2: Detect package installer (uv or pip) ---
USE_UV=false
if command -v uv &>/dev/null; then
    USE_UV=true
    ok "Found uv ($(uv --version 2>/dev/null || echo 'unknown version'))"
else
    warn "uv not found, falling back to pip"
fi

# --- Step 3: Create virtual environment ---
if [[ "$FRESH" == true && -d "$VENV_DIR" ]]; then
    info "Removing existing venv (--fresh)..."
    rm -rf "$VENV_DIR"
    ok "Old venv removed"
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment..."
    if [[ "$USE_UV" == true ]]; then
        uv venv "$VENV_DIR" --python "$PYTHON"
    else
        "$PYTHON" -m venv "$VENV_DIR"
    fi
    ok "Virtual environment created at $VENV_DIR"
else
    ok "Virtual environment already exists at $VENV_DIR"
fi

# Activate venv for the rest of the script
source "$VENV_DIR/bin/activate"

# --- Step 4: Install in editable mode with dev dependencies ---
info "Installing attractor in editable mode with dev dependencies..."
if [[ "$USE_UV" == true ]]; then
    uv pip install -e ".[dev]"
else
    pip install --upgrade pip
    pip install -e ".[dev]"
fi
ok "Installation complete"

# --- Step 5: Verification ---
VERIFY_PASS=true

# 5a: Import check
info "Verifying: import attractor..."
if "$VENV_DIR/bin/python" -c "import attractor; print(f'  version: {attractor.__version__}')" 2>/dev/null; then
    ok "Import check passed"
else
    error "Import check failed"
    VERIFY_PASS=false
fi

# 5b: CLI check
info "Verifying: attractor --help..."
if "$VENV_DIR/bin/attractor" --help &>/dev/null; then
    ok "CLI check passed"
else
    error "CLI check failed — 'attractor --help' returned non-zero"
    VERIFY_PASS=false
fi

# 5c: Test suite
info "Verifying: running test suite..."
if "$VENV_DIR/bin/python" -m pytest --tb=short -q 2>&1; then
    ok "Tests passed"
else
    warn "Some tests failed (see output above)"
    VERIFY_PASS=false
fi

# --- Step 6: Create .env if missing ---
if [[ ! -f "$ENV_FILE" ]]; then
    info "Creating .env with API key placeholders..."
    cat > "$ENV_FILE" << 'ENVEOF'
# Attractor API Keys
# Fill in the keys for the providers you want to use.

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
ENVEOF
    ok "Created $ENV_FILE (fill in your API keys)"
else
    ok ".env already exists"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "  Attractor Install Summary"
echo "============================================"
echo "  Python:     $PYTHON ($("$PYTHON" --version 2>&1))"
echo "  Installer:  $(if $USE_UV; then echo 'uv'; else echo 'pip'; fi)"
echo "  Venv:       $VENV_DIR"
echo "  Fresh:      $FRESH"
echo "  Verified:   $(if $VERIFY_PASS; then echo 'All checks passed'; else echo 'Some checks failed'; fi)"
echo "============================================"
echo ""
echo "  Activate:   source .venv/bin/activate"
echo "  Run:        attractor run"
echo "  Test:       pytest -v"
echo ""

if [[ "$VERIFY_PASS" == false ]]; then
    exit 1
fi
