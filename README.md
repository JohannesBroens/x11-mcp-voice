# Nox — Voice-Activated Desktop Automation

> **⚠️ This is [x11-mcp](https://github.com/JohannesBroens/x11-mcp) with a microphone.** That project gives AI eyes and hands on your desktop. This one gives it ears. You speak a sentence, and a machine that cannot see what you see, cannot hear what you mean, and cannot undo what it does will take control of your mouse and keyboard and act on its best interpretation of your words.
>
> The gap between what you said and what it heard is filled by a speech-to-text model. The gap between what it heard and what it does is filled by a language model. Neither is perfect. Both are confident. And unlike a typed command, you are not looking at a prompt when you speak — you are across the room, or cooking, or not paying attention. The machine acts anyway.
>
> **This is an experimental research project, not production software.** It explores what happens when the interface between human intent and machine action is a spoken sentence — the most natural and the most ambiguous form of instruction we have. Do not run this on a system where unintended actions carry real consequences. The author uses this daily and accepts the risk. You should understand the risk before you do the same.

---

<p align="center">
  <img src="docs/images/nox-somnus.png" width="64" title="somnus — sleeping" alt="Idle">
  <img src="docs/images/nox-excito.png" width="64" title="excito — wake detected" alt="Wake">
  <img src="docs/images/nox-ausculto.png" width="64" title="ausculto — listening" alt="Listening">
  <img src="docs/images/nox-cogito.png" width="64" title="cogito — thinking" alt="Thinking">
  <img src="docs/images/nox-dico.png" width="64" title="dico — speaking" alt="Speaking">
  <img src="docs/images/nox-impero.png" width="64" title="impero — controlling" alt="Controlling">
  <img src="docs/images/nox-erratum.png" width="64" title="erratum — error" alt="Error">
</p>
<p align="center"><sub>somnus &nbsp;&bull;&nbsp; excito &nbsp;&bull;&nbsp; ausculto &nbsp;&bull;&nbsp; cogito &nbsp;&bull;&nbsp; dico &nbsp;&bull;&nbsp; impero &nbsp;&bull;&nbsp; erratum</sub></p>

Say a wake word, speak a command, and Claude controls your desktop — hands-free.

Nox is a voice daemon for Linux that sits on top of [x11-mcp](https://github.com/JohannesBroens/x11-mcp). Where x11-mcp provides the eyes and hands (screen capture, mouse, keyboard), Nox adds the ears and voice — wake word detection, GPU-accelerated speech-to-text, Claude Code for reasoning, and local text-to-speech. Everything runs locally except the Claude call.

## Quick start

```bash
./install.sh     # installs deps, models, systemd units, nox CLI
nox              # start daemon + tray indicator
```

## What it does

1. Listens for a wake word (local, always-on via openwakeword)
2. Transcribes your speech (local, NVIDIA Parakeet, near-instant on CUDA GPUs)
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
