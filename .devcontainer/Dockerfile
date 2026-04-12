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

# Install additional packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
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
        ros-${ROS_DISTRO}-gazebo-ros-control \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge and create a Python 3.8 env with pinocchio (incl. CasADi support)
# Using Python 3.8 keeps ABI compatibility with ROS Noetic's compiled extensions.
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -n robot python=3.8 pinocchio -c conda-forge -y \
    && /opt/conda/bin/conda clean -afy
ENV PATH="/opt/conda/envs/robot/bin:${PATH}"

# Install remaining Python dependencies into the conda env
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Expose ROS Noetic and system Python packages to the conda Python 3.8 env only.
# Using an activation hook avoids polluting the global PYTHONPATH, which breaks
# the conda CLI (base Python 3.13) due to incompatible system urllib3/six.
RUN mkdir -p /opt/conda/envs/robot/etc/conda/activate.d \
    && echo 'export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:${PYTHONPATH}"' \
       > /opt/conda/envs/robot/etc/conda/activate.d/ros_pythonpath.sh

ENV SHELL /bin/bash

# Source ROS environment automatically
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /home/ws/devel/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc \
    && echo "export GAZEBO_PLUGIN_PATH=\$GAZEBO_PLUGIN_PATH:/usr/local/lib/gazebo-11/plugins" >> /home/$USERNAME/.bashrc \
    && echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:/home/ws/src" >> /home/$USERNAME/.bashrc

# Set the default user
USER $USERNAME
CMD ["/bin/bash"]
