# ROS2 Humble + CARLA Python client for autonomous navigation and traffic sign detection
FROM ros:humble-ros-base-jammy

# ── Environment ────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    CARLA_HOST=localhost \
    CARLA_PORT=2001

# ── System + ROS2 packages ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python base
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-pygame \
    python3-networkx \
    # ROS2 build toolchain
    python3-colcon-common-extensions \
    python3-rosdep \
    # OpenGL / display
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libxrender1 \
    libxi6 \
    libxmu6 \
    # General tools
    wget \
    curl \
    git \
    cmake \
    build-essential \
    software-properties-common \
    # ROS2 core + bridge dependencies
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-foxglove-bridge \
    ros-${ROS_DISTRO}-foxglove-msgs \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    # carla-ros-bridge runtime deps
    ros-${ROS_DISTRO}-ackermann-msgs \
    ros-${ROS_DISTRO}-derived-object-msgs \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-tf2-eigen \
    ros-${ROS_DISTRO}-rqt \
    ros-${ROS_DISTRO}-python-qt-binding \
    && rm -rf /var/lib/apt/lists/*

# ── CARLA Python client ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir carla==0.9.15 transforms3d

# ── carla-ros-bridge (source, compatible with CARLA 0.9.15) ─────────────────
RUN git clone --recurse-submodules \
    https://github.com/carla-simulator/ros-bridge.git \
    /home/ros/workspace/src/ros-bridge \
 && echo "0.9.15" > \
    /home/ros/workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION \
 && find /home/ros/workspace/src/ros-bridge -name "*.py" -exec \
    sed -i 's/\bnumpy\.bool\b/numpy.bool_/g; s/\bnp\.bool\b/np.bool_/g' {} +

# ── Workspace ─────────────────────────────────────────────────────────────────
RUN mkdir -p /home/ros/workspace/src
WORKDIR /home/ros/workspace

# ── Use bash so 'source' works in RUN steps ───────────────────────────────────
SHELL ["/bin/bash", "-c"]

# ── Pre-build ros-bridge packages ─────────────────────────────────────────────
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
 && colcon build --symlink-install --packages-skip pcl_recorder rviz_carla_plugin carla_ad_demo

# ── Shell setup ───────────────────────────────────────────────────────────────
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
 && echo "[ -f /home/ros/workspace/install/setup.bash ] && source /home/ros/workspace/install/setup.bash" >> /root/.bashrc

# Default command
CMD ["bash"]