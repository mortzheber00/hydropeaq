#!/usr/bin/env bash
# Runs inside the container after it is created.
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
NRP_GAZEBO_DIR="/home/ws/nrp_gazebo"
SPLISHSPLASH_DIR="/home/ws/splishsplash"
CATKIN_WS="/home/ws"
LOCAL_PREFIX="$HOME/.local"
SPH_OUTPUT_DIR="/home/ws/sph_output"

# ── Bypass conda for native C++ builds ───────────────────────────────────────
# The Dockerfile prepends the conda env to PATH globally; cmake must link
# against system libraries and headers, not conda's incompatible variants.
# We pass an explicit SYSTEM_PATH to every cmake/make invocation below.
SYSTEM_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

mkdir -p "$LOCAL_PREFIX/bin" "$LOCAL_PREFIX/lib"

# ── 1. Clone and build NRP Gazebo ─────────────────────────────────────────────
if [ ! -f "$LOCAL_PREFIX/bin/gazebo" ]; then
    if [ ! -d "$NRP_GAZEBO_DIR/.git" ]; then
        echo "==> Cloning NRP Gazebo (development branch)..."
        git clone --depth 1 \
            --branch development \
            https://bitbucket.org/hbpneurorobotics/gazebo.git \
            "$NRP_GAZEBO_DIR"
    else
        echo "==> NRP Gazebo already cloned, skipping clone."
    fi

    # Fix ownership in case a previous run cloned/built as root
    sudo chown -R "$(id -u):$(id -g)" "$NRP_GAZEBO_DIR"

    echo "==> Building NRP Gazebo (this takes a while)..."
    mkdir -p "$NRP_GAZEBO_DIR/build"
    cd "$NRP_GAZEBO_DIR/build"
    env PATH="$SYSTEM_PATH" cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$LOCAL_PREFIX"
    env PATH="$SYSTEM_PATH" make -j"$(nproc)"
    env PATH="$SYSTEM_PATH" make install
    echo "==> NRP Gazebo installed to $LOCAL_PREFIX"
else
    echo "==> NRP Gazebo already installed at $LOCAL_PREFIX/bin/gazebo, skipping."
fi

# Activate $HOME/.local for the remainder of this script
export PATH="$LOCAL_PREFIX/bin:$SYSTEM_PATH"
export LD_LIBRARY_PATH="$LOCAL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GAZEBO_PLUGIN_PATH="$LOCAL_PREFIX/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"

# ── 2. Clone and build HBP SPlisHSPlasH (Gazebo fluid plugin) ────────────────
# Per: https://bitbucket.org/hbpneurorobotics/neurorobotics-platform/src/master/fluid_simulation_install.md
if [ ! -d "$SPLISHSPLASH_DIR/.git" ]; then
    echo "==> Cloning HBP SPlisHSPlasH..."
    git clone \
        https://bitbucket.org/hbpneurorobotics/splishsplash.git \
        "$SPLISHSPLASH_DIR"
else
    echo "==> SPlisHSPlasH already cloned, skipping."
fi

# Fix ownership in case a previous run cloned/built as root
sudo chown -R "$(id -u):$(id -g)" "$SPLISHSPLASH_DIR"

echo "==> Building SPlisHSPlasH + Gazebo fluid plugin..."
mkdir -p "$SPLISHSPLASH_DIR/build"
cd "$SPLISHSPLASH_DIR/build"
# gazebo_DIR must point explicitly to the NRP cmake config so find_package(gazebo)
# does not fall back to the system Gazebo in /usr/lib/x86_64-linux-gnu/cmake/gazebo.
env PATH="$SYSTEM_PATH" cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_DOUBLE_PRECISION=OFF \
    -DBUILD_GAZEBO_PLUGIN=ON \
    -DCMAKE_PREFIX_PATH="$LOCAL_PREFIX" \
    -Dgazebo_DIR="$LOCAL_PREFIX/lib/cmake/gazebo" \
    -DEIGEN3_VERSION_STRING=3.3.7 \
    -DCMAKE_INSTALL_PREFIX="$LOCAL_PREFIX"
env PATH="$SYSTEM_PATH" make -j"$(nproc)"
env PATH="$SYSTEM_PATH" make install
echo "==> SPlisHSPlasH installed to $LOCAL_PREFIX"

# ── 3. Create SPH output and boundary mesh directories ─────────────────────────
mkdir -p "$SPH_OUTPUT_DIR"
echo "==> SPH output directory: $SPH_OUTPUT_DIR"
mkdir -p /home/ws/sph_boundaries
echo "==> SPH boundary mesh directory: /home/ws/sph_boundaries"

# ── 4. Verify NRP Gazebo is the active gazebo ──────────────────────────────────
GAZEBO_BIN="$(PATH="$LOCAL_PREFIX/bin:$SYSTEM_PATH" which gazebo)"
echo "==> Active gazebo binary: $GAZEBO_BIN"
if [ "$GAZEBO_BIN" != "$LOCAL_PREFIX/bin/gazebo" ]; then
    echo "WARNING: expected $LOCAL_PREFIX/bin/gazebo but got $GAZEBO_BIN"
fi

# ── 5. Build the catkin workspace ──────────────────────────────────────────────
# Restore full PATH (including conda) so catkin can find ROS Python tools.
export PATH="$LOCAL_PREFIX/bin:$PATH"
cd "$CATKIN_WS"
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release

echo ""
echo "==> Done. To run the fluid simulation:"
echo "    gazebo '/home/ws/BoxDamBreak (3).sdf' -g libFluidVisPlugin.so"
echo "    (output particles will be written to $SPH_OUTPUT_DIR)"
