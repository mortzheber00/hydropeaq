#!/usr/bin/env bash
# Runs inside the container after it is created.
set -euo pipefail

SPLISHSPLASH_DIR="/home/ws/splishsplash"
CATKIN_WS="/home/ws"

# ── 1. Clone and build HBP SPlisHSPlasH (Gazebo plugin) ─────────────────────
if [ ! -d "$SPLISHSPLASH_DIR" ]; then
    echo "==> Cloning HBP SPlisHSPlasH..."
    git clone https://bitbucket.org/hbpneurorobotics/splishsplash.git "$SPLISHSPLASH_DIR"
else
    echo "==> SPlisHSPlasH already cloned, skipping."
fi

echo "==> Building SPlisHSPlasH + Gazebo plugin..."
mkdir -p "$SPLISHSPLASH_DIR/build"
cd "$SPLISHSPLASH_DIR/build"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_DOUBLE_PRECISION=OFF \
    -DBUILD_GAZEBO_PLUGIN=ON \
    -DGAZEBO_DIR=/usr/share/gazebo-11
make -j"$(nproc)"
sudo make install

# ── 2. Build the catkin workspace ─────────────────────────────────────────────
# The amph package lives at src/amph (package.xml + CMakeLists.txt).
# /home/ws is the workspace root; catkin_make is invoked from there.

cd "$CATKIN_WS"
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release

echo ""
echo "==> Done. Reload your shell or run:"
echo "    source /home/ws/devel/setup.bash"
