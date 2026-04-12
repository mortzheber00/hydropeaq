#!/usr/bin/env sh
# Runs on the HOST before the container is created.
# Grants the container access to the host X11 server.
xhost +local:docker
