ARG DOCKER_REPO
ARG ROS_DISTRO
ARG IMAGE_SUFFIX
FROM $DOCKER_REPO:$ROS_DISTRO$IMAGE_SUFFIX

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Redeclare so it is available below the FROM line
ARG ROS_DISTRO

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install additional packages (apt-get upgrade removed for build reproducibility)
RUN apt-get update && \
    apt-get install -y --fix-missing \
        python3-pip \
        python-is-python3 \
        python3-catkin-tools \
        jq \
        ssh \
        neovim \
        git \
        curl \
        ros-${ROS_DISTRO}-ros-control \
        ros-${ROS_DISTRO}-ros-controllers \
        ros-${ROS_DISTRO}-polled-camera \
        ros-${ROS_DISTRO}-camera-info-manager \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies for NRP Gazebo and SPlisHSPlasH.
# These are compiled outside conda to avoid ABI conflicts.
RUN apt-get update && apt-get install -y --fix-missing \
        build-essential \
        cmake \
        pkg-config \
        mercurial \
        libboost-all-dev \
        libgts-dev \
        libgdal-dev \
        libltdl-dev \
        libxml2-dev \
        qtbase5-dev \
        libqwt-qt5-dev \
        libbullet-dev \
        libogre-1.9-dev \
        libfreeimage-dev \
        protobuf-compiler \
        libprotobuf-dev \
        libtar-dev \
        libtbb-dev \
        libcurl4-openssl-dev \
        libbz2-dev \
        uuid-dev \
        libavcodec-dev \
        libavformat-dev \
        libavdevice-dev \
        libswscale-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libignition-math6-dev \
        libignition-transport8-dev \
        libignition-msgs5-dev \
        libignition-common3-dev \
        libignition-fuel-tools4-dev \
        libsdformat9-dev \
        libccd-dev \
        libusb-1.0-0-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit for SPlisHSPlasH GPU neighborhood search (cuNSearch).
# Match cuda-toolkit-XX-X to your host NVIDIA driver — check with: nvidia-smi
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends cuda-toolkit-12-6 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda-12.6/bin:${PATH}"
ENV CUDA_HOME="/usr/local/cuda-12.6"

# Install Miniforge and create a Python 3.8 env with pinocchio (incl. CasADi support)
# Using Python 3.8 keeps ABI compatibility with ROS Noetic's compiled extensions.
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -n robot python=3.8 pinocchio -c conda-forge -y \
    && /opt/conda/bin/conda clean -afy

# Install remaining Python dependencies into the conda env
COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/robot/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Expose ROS Noetic and system Python packages to the conda Python 3.8 env only.
# A .pth file in site-packages is scoped to that interpreter — unlike PYTHONPATH
# it never bleeds into the conda base Python 3.13, so the conda CLI keeps working.
RUN /opt/conda/envs/robot/bin/python - <<'EOF'
import site, pathlib
pth = pathlib.Path(site.getsitepackages()[0]) / "ros_system.pth"
pth.write_text("/opt/ros/noetic/lib/python3/dist-packages\n/usr/lib/python3/dist-packages\n")
EOF

ENV SHELL /bin/bash

# Initialise conda shell functions and auto-activate the robot env.
RUN echo '. /opt/conda/etc/profile.d/conda.sh' >> /home/$USERNAME/.bashrc \
    && echo "conda activate robot" >> /home/$USERNAME/.bashrc

# Source ROS environment automatically and expose NRP Gazebo / SPlisHSPlasH
# installed to $HOME/.local (built in postCreate, outside conda).
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /home/ws/devel/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc \
    && echo "export PATH=\$HOME/.local/bin:\$PATH" >> /home/$USERNAME/.bashrc \
    && echo "export LD_LIBRARY_PATH=\$HOME/.local/lib:\$LD_LIBRARY_PATH" >> /home/$USERNAME/.bashrc \
    && echo "export PKG_CONFIG_PATH=\$HOME/.local/lib/pkgconfig:\$PKG_CONFIG_PATH" >> /home/$USERNAME/.bashrc \
    && echo "export GAZEBO_PLUGIN_PATH=\$HOME/.local/lib/gazebo-11/plugins:\$HOME/.local/lib:\$GAZEBO_PLUGIN_PATH:/usr/local/lib/gazebo-11/plugins" >> /home/$USERNAME/.bashrc \
    && echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:/home/ws/src" >> /home/$USERNAME/.bashrc \
    && echo '[ -f $HOME/.local/share/gazebo/setup.sh ] && source $HOME/.local/share/gazebo/setup.sh' >> /home/$USERNAME/.bashrc

# Set the default user
USER $USERNAME
CMD ["/bin/bash"]
