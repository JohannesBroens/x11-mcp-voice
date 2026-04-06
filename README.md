# x11-mcp-voice

Voice-activated desktop automation for Linux.

Say a wake word, speak a command, and Claude controls your desktop — hands-free.

## What it does

1. Listens for a wake word (local, always-on)
2. Transcribes your speech (local, GPU-accelerated)
3. Claude executes desktop actions via [x11-mcp](https://github.com/JohannesBroens/x11-mcp)
4. Speaks the response back to you (local TTS)

Everything runs locally except the Claude call.

## Requirements

- Linux with X11
- NVIDIA GPU with CUDA
- [Claude Code](https://claude.ai/download) with an active subscription
- [x11-mcp](https://github.com/JohannesBroens/x11-mcp) set up as a Claude Code MCP server

## Quick start

```bash
./install.sh            # installs deps, downloads models, registers MCP
source .venv/bin/activate
python -m x11_mcp_voice -v
```

## Configuration

```bash
cp config.example.yaml config.yaml
# edit to taste — all fields have sensible defaults
```

## License

MIT
