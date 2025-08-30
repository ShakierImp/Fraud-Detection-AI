#!/bin/bash
# =====================================================================
# FraudGuardian AI Integration Script
# =====================================================================
# This script sets up and runs the FraudGuardian AI project for a local
# demo. It automates environment setup, dependency installation,
# backend API launch, and frontend dashboard launch.
#
# ---------------------------------------------------------------------
# For Windows Users (Git Bash / WSL):
# ---------------------------------------------------------------------
# This script is designed for Unix-like systems (Linux, macOS).
# On Windows, please use Git Bash or Windows Subsystem for Linux (WSL).
# Alternatively, run these commands manually in PowerShell or CMD:
#
#   python -m venv venv
#   .\venv\Scripts\activate
#   pip install -r requirements.txt
#   uvicorn src.api.inference_api_fast:app --port 8000 --host 0.0.0.0
#   streamlit run src/frontend/dashboard_streamlit.py
#
# =====================================================================

# --------------------------
# Default variables
# --------------------------
MODE="local"
VENV_DIR="venv"
BACKEND_PID=""

# --------------------------
# Usage function
# --------------------------
usage() {
    echo "Usage: $0 [--local|--docker|--help]"
    echo
    echo "Options:"
    echo "  --local   Run in local mode (default). Sets up Python venv, installs deps, runs backend + Streamlit."
    echo "  --docker  Placeholder: Not implemented yet."
    echo "  --help    Show this help message."
    exit 0
}

# --------------------------
# Argument parsing
# --------------------------
for arg in "$@"; do
    case $arg in
        --local)
            MODE="local"
            shift
            ;;
        --docker)
            echo "Docker mode is not implemented yet. Please use --local."
            exit 1
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            ;;
    esac
done

# --------------------------
# Check for python
# --------------------------
check_python() {
    echo "🔎 Checking for Python..."
    if command -v python3 &>/dev/null; then
        PYTHON_BIN="python3"
    elif command -v python &>/dev/null; then
        PYTHON_BIN="python"
    else
        echo "❌ Python is not installed. Please install Python 3."
        exit 1
    fi
    echo "✅ Found Python: $($PYTHON_BIN --version)"
}

# --------------------------
# Create virtual environment
# --------------------------
create_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "📦 Creating virtual environment in $VENV_DIR..."
        $PYTHON_BIN -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "❌ Failed to create virtual environment."
            exit 1
        fi
    else
        echo "ℹ️  Virtual environment already exists. Skipping creation."
    fi
}

# --------------------------
# Activate virtual environment
# --------------------------
activate_venv() {
    echo "⚡ Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate virtual environment."
        exit 1
    fi
}

# --------------------------
# Install dependencies
# --------------------------
install_deps() {
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies."
        exit 1
    fi
}

# --------------------------
# Start backend API
# --------------------------
start_backend() {
    echo "🚀 Starting FastAPI backend..."
    uvicorn src.api.inference_api_fast:app --port 8000 --host 0.0.0.0 &
    BACKEND_PID=$!
    echo "ℹ️  Backend PID: $BACKEND_PID"
}

# --------------------------
# Start frontend dashboard
# --------------------------
start_frontend() {
    echo "🚀 Starting Streamlit dashboard..."
    streamlit run src/frontend/dashboard_streamlit.py
}

# --------------------------
# Trap cleanup
# --------------------------
cleanup() {
    echo
    echo "🛑 Caught exit signal. Cleaning up..."
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "🔪 Killing backend process $BACKEND_PID..."
        kill "$BACKEND_PID"
    fi
    exit 0
}

trap cleanup EXIT SIGINT

# --------------------------
# Main execution
# --------------------------
if [ "$MODE" == "local" ]; then
    check_python
    create_venv
    activate_venv
    install_deps

    echo "🌐 Starting services..."
    echo "🔧 FastAPI API will be at: http://localhost:8000"
    echo "📊 API Documentation will be at: http://localhost:8000/docs"
    echo "📈 Streamlit Dashboard will be at: http://localhost:8501"

    start_backend
    start_frontend
fi
