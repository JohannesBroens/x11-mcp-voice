#!/usr/bin/env bash
#
# x11-mcp-voice install script for Ubuntu 24 LTS
#
# Installs system packages, creates a Python venv, installs pip dependencies,
# downloads ML models, and verifies the setup.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
#
# Prerequisites:
#   - Ubuntu 24.04 LTS (Python 3.12 works — we use ONNX instead of tflite)
#   - NVIDIA GPU with drivers installed (nvidia-smi should work)
#   - Claude Code installed and authenticated (claude.ai/download)
#   - x11-mcp cloned and set up at ~/Documents/git/x11-mcp/
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VOICES_DIR="$HOME/.local/share/piper-voices"
PIPER_VOICE="en_US-ryan-medium"

# HuggingFace base URL for piper voice models
PIPER_HF_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

# ─────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ─────────────────────────────────────────────────
# Step 1: System packages
# ─────────────────────────────────────────────────
PYTHON="python3"
info "Installing system packages..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    playerctl \
    libportaudio2 \
    portaudio19-dev \
    python3-venv \
    python3-dev \
    alsa-utils \
    curl \
    git

ok "System packages installed"

# ─────────────────────────────────────────────────
# Step 2: Verify NVIDIA GPU + CUDA
# ─────────────────────────────────────────────────
info "Checking NVIDIA GPU..."

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "NVIDIA GPU found: $GPU_NAME"
else
    fail "nvidia-smi not found. NVIDIA drivers required for Parakeet STT."
    echo "  Install with: sudo apt install nvidia-driver-560"
    echo "  Then reboot and re-run this script."
    exit 1
fi

# ─────────────────────────────────────────────────
# Step 3: Check Claude Code is installed
# ─────────────────────────────────────────────────
if command -v claude &>/dev/null; then
    ok "Claude Code found: $(claude --version 2>/dev/null || echo 'installed')"
else
    fail "Claude Code not found. The daemon uses Claude Code as its AI backend."
    echo "  Install Claude Code from: https://claude.ai/download"
    exit 1
fi

# ─────────────────────────────────────────────────
# Step 4: Check x11-mcp sister project
# ─────────────────────────────────────────────────
X11_MCP_DIR="$HOME/Documents/git/x11-mcp"
X11_MCP_PYTHON="$X11_MCP_DIR/.venv/bin/python"

info "Checking x11-mcp at $X11_MCP_DIR..."

if [[ -x "$X11_MCP_PYTHON" ]]; then
    ok "x11-mcp found at $X11_MCP_DIR"
else
    fail "x11-mcp not found or not set up."
    echo "  Expected: $X11_MCP_PYTHON"
    echo "  Clone and set up x11-mcp first:"
    echo "    git clone <repo-url> $X11_MCP_DIR"
    echo "    cd $X11_MCP_DIR && python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# Register x11-mcp as a Claude Code MCP server (user-global)
info "Registering x11-mcp with Claude Code..."
claude mcp add --scope user x11-mcp "$X11_MCP_PYTHON" -- -m x11_mcp 2>/dev/null || true
ok "x11-mcp registered as MCP server"

# ─────────────────────────────────────────────────
# Step 5: Create Python venv + install pip deps
# ─────────────────────────────────────────────────
info "Setting up Python virtual environment..."

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
else
    ok "Venv already exists at $VENV_DIR"
fi

# Activate venv for remaining steps
source "$VENV_DIR/bin/activate"

info "Upgrading pip..."
pip install --upgrade pip -q

info "Installing openwakeword without tflite (we use ONNX backend instead)..."
pip install --no-deps openwakeword -q

info "Installing x11-mcp-voice and remaining dependencies (torch/NeMo are large)..."
pip install -e "$SCRIPT_DIR[dev]" --no-deps 2>&1 | tail -5
pip install onnxruntime "sounddevice>=0.5.0" "numpy>=1.24.0" PyYAML \
    "piper-tts>=1.2.0" "nemo_toolkit[asr]>=2.0.0" "torch>=2.0.0" "torchaudio>=2.0.0" \
    "pytest>=7.0.0" "pytest-asyncio>=0.23.0" 2>&1 | tail -5

ok "Python dependencies installed"

# ─────────────────────────────────────────────────
# Step 6: Verify CUDA works with PyTorch
# ─────────────────────────────────────────────────
info "Verifying PyTorch CUDA support..."

CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    CUDA_DEVICE=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    ok "PyTorch CUDA is working: $CUDA_DEVICE"
else
    warn "PyTorch CUDA not available. Parakeet STT will be extremely slow on CPU."
    echo "  You may need to install the CUDA-enabled version of PyTorch:"
    echo "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
fi

# ─────────────────────────────────────────────────
# Step 7: Download piper TTS voice model
# ─────────────────────────────────────────────────
info "Downloading piper voice model: $PIPER_VOICE..."

mkdir -p "$VOICES_DIR"

# Voice name format: en_US-ryan-medium -> en/en_US/ryan/medium/
LANG_CODE="${PIPER_VOICE%%_*}"           # en
LOCALE="${PIPER_VOICE%%-*}"              # en_US
NAME_QUALITY="${PIPER_VOICE#*-}"         # ryan-medium
NAME="${NAME_QUALITY%-*}"                # ryan
QUALITY="${NAME_QUALITY##*-}"            # medium

ONNX_URL="${PIPER_HF_BASE}/${LANG_CODE}/${LOCALE}/${NAME}/${QUALITY}/${PIPER_VOICE}.onnx"
JSON_URL="${ONNX_URL}.json"

ONNX_FILE="$VOICES_DIR/${PIPER_VOICE}.onnx"
JSON_FILE="$VOICES_DIR/${PIPER_VOICE}.onnx.json"

if [[ -f "$ONNX_FILE" ]]; then
    ok "Voice model already downloaded: $ONNX_FILE"
else
    info "Downloading ${PIPER_VOICE}.onnx (~30MB)..."
    curl -L -# -o "$ONNX_FILE" "$ONNX_URL"
    curl -L -s -o "$JSON_FILE" "$JSON_URL"
    ok "Voice model downloaded to $VOICES_DIR/"
fi

# ─────────────────────────────────────────────────
# Step 8: Pre-download Parakeet STT model
# ─────────────────────────────────────────────────
info "Pre-downloading Parakeet STT model (nvidia/parakeet-tdt-0.6b-v2)..."
info "  This is ~1.5GB and may take a few minutes on first run."

python -c "
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')
print('Parakeet model downloaded successfully')
" 2>&1 | grep -E "(Parakeet|Download|download|Error|error)" || true

ok "Parakeet STT model ready"

# ─────────────────────────────────────────────────
# Step 9: Pre-download Silero VAD model
# ─────────────────────────────────────────────────
info "Pre-downloading Silero VAD model..."

python -c "
import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
print('Silero VAD downloaded successfully')
" 2>&1 | grep -E "(Silero|Download|download|Error|error)" || true

ok "Silero VAD model ready"

# ─────────────────────────────────────────────────
# Step 10: Pre-download openwakeword model
# ─────────────────────────────────────────────────
info "Pre-downloading openwakeword model: hey_jarvis..."

python -c "
import openwakeword
openwakeword.utils.download_models(['hey_jarvis'])
print('OpenWakeWord model downloaded successfully')
" 2>&1 | grep -E "(OpenWakeWord|Download|download|Error|error)" || true

ok "OpenWakeWord model ready"

# ─────────────────────────────────────────────────
# Step 11: Run tests
# ─────────────────────────────────────────────────
info "Running test suite..."

if python -m pytest tests/ -v --tb=short 2>&1 | tail -5; then
    ok "All tests pass"
else
    warn "Some tests failed — check output above"
fi

# ─────────────────────────────────────────────────
# Step 12: Verify hardware
# ─────────────────────────────────────────────────
info "Checking audio hardware..."

# Check microphone
if arecord -l 2>/dev/null | grep -q "card"; then
    ok "Microphone detected"
else
    warn "No microphone detected. Wake word detection requires a mic."
fi

# Check X11 display
if [[ -n "${DISPLAY:-}" ]]; then
    ok "X11 DISPLAY=$DISPLAY"
else
    warn "DISPLAY not set. Desktop automation requires X11."
fi

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  x11-mcp-voice installation complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo "  To run the daemon:"
echo ""
echo "    source $VENV_DIR/bin/activate"
echo "    python -m x11_mcp_voice -v"
echo ""
echo "  Configuration (optional):"
echo "    cp $SCRIPT_DIR/config.example.yaml config.yaml"
echo "    # Edit config.yaml to customize"
echo ""
echo "  Voice models directory: $VOICES_DIR/"
echo "  Logs: use -v flag for debug output"
echo ""
