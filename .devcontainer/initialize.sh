#!/usr/bin/env sh
# Runs on the host before the container is created.
# Allows the container to use the host X server (skipped without a display).
if [ -n "$DISPLAY" ]; then
    xhost +local:docker
else
    echo "No DISPLAY set, skipping xhost (GUI will not be available)"
fi
