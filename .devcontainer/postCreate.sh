#!/usr/bin/env bash
# Runs inside the container after it is created.
set -euo pipefail

# --- Paths ---
NRP_GAZEBO_DIR="/home/ws/nrp_gazebo"
NRP_GAZEBO_ROS_DIR="/home/ws/src/gazebo_ros_pkgs"
SPLISHSPLASH_DIR="/home/ws/splishsplash"
CATKIN_WS="/home/ws"
LOCAL_PREFIX="$HOME/.local"
SPH_OUTPUT_DIR="/home/ws/sph_output"

# --- Native builds without conda ---
# conda is first on PATH; C++ builds must use the system toolchain and libraries.
SYSTEM_PATH="/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

mkdir -p "$LOCAL_PREFIX/bin" "$LOCAL_PREFIX/lib"

# --- 1. NRP Gazebo ---
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

    # A previous run may have built as root
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

# Use $HOME/.local for the rest of the script
export PATH="$LOCAL_PREFIX/bin:$SYSTEM_PATH"
export LD_LIBRARY_PATH="$LOCAL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$LOCAL_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export GAZEBO_PLUGIN_PATH="$LOCAL_PREFIX/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"

# --- 2. NRP gazebo_ros_pkgs ---
# Cloned into the workspace and built against NRP Gazebo in step 6.
if [ ! -d "$NRP_GAZEBO_ROS_DIR/.git" ]; then
    echo "==> Cloning NRP gazebo_ros_pkgs (development branch)..."
    git clone --depth 1 \
        --branch development \
        https://bitbucket.org/hbpneurorobotics/gazeborospackages.git \
        "$NRP_GAZEBO_ROS_DIR"
else
    echo "==> NRP gazebo_ros_pkgs already cloned, skipping."
fi
sudo chown -R "$(id -u):$(id -g)" "$NRP_GAZEBO_ROS_DIR"

# --- 3. SPlisHSPlasH and the Gazebo fluid plugin ---
echo "==> Building SPlisHSPlasH + Gazebo fluid plugin..."
mkdir -p "$SPLISHSPLASH_DIR/build"
cd "$SPLISHSPLASH_DIR/build"
# gazebo_DIR forces NRP Gazebo instead of a system install.
env PATH="$SYSTEM_PATH" cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_DOUBLE_PRECISION=OFF \
    -DBUILD_GAZEBO_PLUGIN=ON \
    -DCMAKE_PREFIX_PATH="$LOCAL_PREFIX" \
    -Dgazebo_DIR="$LOCAL_PREFIX/lib/cmake/gazebo" \
    -DEIGEN3_VERSION_STRING=3.3.7 \
    -DCMAKE_INSTALL_PREFIX="$LOCAL_PREFIX" \
    -DUSE_OpenMP=ON \
    -DUSE_GPU_NEIGHBORHOOD_SEARCH=OFF \
    -DUSE_AVX=ON \
    -DCMAKE_CUDA_COMPILER="$(find /usr/local/cuda*/bin -name nvcc 2>/dev/null | head -1)"

env PATH="$SYSTEM_PATH" make -j"$(nproc)"
env PATH="$SYSTEM_PATH" make install
echo "==> SPlisHSPlasH installed to $LOCAL_PREFIX"

# --- 4. SPH output and boundary mesh directories ---
mkdir -p "$SPH_OUTPUT_DIR"
echo "==> SPH output directory: $SPH_OUTPUT_DIR"
mkdir -p /home/ws/sph_boundaries
echo "==> SPH boundary mesh directory: /home/ws/sph_boundaries"

# --- 5. Check that NRP Gazebo is the active gazebo ---
GAZEBO_BIN="$(PATH="$LOCAL_PREFIX/bin:$SYSTEM_PATH" which gazebo)"
echo "==> Active gazebo binary: $GAZEBO_BIN"
if [ "$GAZEBO_BIN" != "$LOCAL_PREFIX/bin/gazebo" ]; then
    echo "WARNING: expected $LOCAL_PREFIX/bin/gazebo but got $GAZEBO_BIN"
fi

# --- 6. Catkin workspace ---
# Full PATH again (with conda) for the ROS Python tools; gazebo_DIR as in step 3.
export PATH="$LOCAL_PREFIX/bin:$PATH"
cd "$CATKIN_WS"
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
# Only the needed packages; the NRP repo has many plugins with missing dependencies.
catkin_make \
    --only-pkg-with-deps amph gazebo_dev gazebo_msgs gazebo_plugins gazebo_ros gazebo_ros_control \
    --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -Dgazebo_DIR="$LOCAL_PREFIX/lib/cmake/gazebo"

echo ""
echo "==> Done. To run the fluid simulation:"
echo "    gazebo '/home/ws/BoxDamBreak (3).sdf' -g libFluidVisPlugin.so"
echo "    (output particles will be written to $SPH_OUTPUT_DIR)"
