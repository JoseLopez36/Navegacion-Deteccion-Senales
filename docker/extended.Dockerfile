# Extended image: ROS2 Humble + CARLA on top of CUDA/TensorFlow base
# Build: docker build -f extended.Dockerfile -t navdet-extended .
ARG BASE_IMAGE=navdet-base:latest
FROM ${BASE_IMAGE}

# ── ROS2 Environment ───────────────────────────────────────────────────────────
ENV ROS_DISTRO=humble \
    CARLA_HOST=localhost \
    CARLA_PORT=2001 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# ── Add ROS2 repository ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && rm -rf /var/lib/apt/lists/*

# ── Install ROS2 Humble base + tools ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-argcomplete \
    python3-numpy \
    python3-opencv \
    python3-pygame \
    python3-networkx \
    && rm -rf /var/lib/apt/lists/*

# ── ROS2 vision/robotics packages ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# ── CARLA Python client ──────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir carla==0.9.15 transforms3d

# ── carla-ros-bridge (source, compatible with CARLA 0.9.15) ─────────────────────
RUN mkdir -p /home/ros/workspace/src \
    && git clone --recurse-submodules \
       https://github.com/carla-simulator/ros-bridge.git \
       /home/ros/workspace/src/ros-bridge \
    && echo "0.9.15" > /home/ros/workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION \
    && find /home/ros/workspace/src/ros-bridge -name "*.py" -exec \
       sed -i 's/\bnumpy\.bool\b/numpy.bool_/g; s/\bnp\.bool\b/np.bool_/g' {} +

# ── Workspace setup ────────────────────────────────────────────────────────────
WORKDIR /home/ros/workspace

# ── Use bash so 'source' works in RUN steps ────────────────────────────────────
SHELL ["/bin/bash", "-c"]

# ── Initialize rosdep and install dependencies ─────────────────────────────────
RUN rosdep init || true \
    && rosdep update --rosdistro ${ROS_DISTRO} || true \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && rosdep install --from-paths src --ignore-src -r -y || true

# ── Build ros-bridge (without --symlink-install to avoid setuptools issues) ───
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --packages-skip pcl_recorder rviz_carla_plugin carla_ad_demo

# ── Shell setup ───────────────────────────────────────────────────────────────
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc \
    && echo "[ -f /home/ros/workspace/install/setup.bash ] && source /home/ros/workspace/install/setup.bash" >> /root/.bashrc

# Default command
CMD ["bash"]