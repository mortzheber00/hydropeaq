#!/usr/bin/env sh
# Runs on the HOST before the container is created.
# Grants the container access to the host X11 server (skipped if no display, e.g. SSH without X forwarding).
if [ -n "$DISPLAY" ]; then
    xhost +local:docker
else
    echo "No DISPLAY set, skipping xhost (GUI will not be available)"
fi
