#!/usr/bin/env bash
set -e

# System dependencies for transcriber + dictate.py
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    wmctrl \
    xdotool \
    xclip \
    libportaudio2 \
    portaudio19-dev

echo "Done. Optional extras:"
echo "  wtype  — Wayland paste support: sudo apt install wtype"
echo ""
echo "If running in a conda environment and sounddevice fails with a GLIBCXX error, run:"
echo "  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 python dictate.py"
