# Nox — Voice-Activated Desktop Automation

Say a wake word, speak a command, and Claude controls your desktop — hands-free.

Nox is a voice daemon for Linux that chains wake word detection, GPU-accelerated speech-to-text, Claude Code, [x11-mcp](https://github.com/JohannesBroens/x11-mcp) desktop automation, and local text-to-speech into a single hands-free pipeline. Everything runs locally except the Claude call.

## Quick start

```bash
./install.sh     # installs deps, models, systemd units, nox CLI
nox              # start daemon + tray indicator
```

## What it does

1. Listens for a wake word (local, always-on via openwakeword)
2. Transcribes your speech (local, NVIDIA Parakeet, ~10ms on NVIDIA GPU)
3. Claude executes desktop actions via [x11-mcp](https://github.com/JohannesBroens/x11-mcp)
4. Speaks the response back to you (local piper-tts)
5. System tray icon shows live state with expressive Nox faces

## Nox CLI

```bash
nox              # start daemon + tray (checks if already running)
nox status       # show current state (somnus, ausculto, cogito, etc.)
nox log          # tail ~/.local/log/nox/daemon.log
nox stop         # stop daemon + tray
nox install      # enable systemd autostart
nox uninstall    # disable autostart
```

## Requirements

- Linux with X11 (Ubuntu 24.04 LTS tested)
- NVIDIA GPU with CUDA
- [Claude Code](https://claude.ai/download) with an active subscription
- [x11-mcp](https://github.com/JohannesBroens/x11-mcp) set up as a Claude Code MCP server

## Configuration

```bash
cp config.example.yaml config.yaml
# edit to taste — all fields have sensible defaults
```

## License

MIT
