# Push-Alphabet simulation image.
# Base: the lecture image (Ubuntu 22.04 + ROS 2 Humble desktop + Gazebo Classic 11
# + MATE desktop over noVNC). We only add what the project needs on top.
FROM airobotbook/ros2-desktop-ai-robot-book-humble:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# The ROS apt key shipped in older images expired (2025-06); refresh it so that
# `apt-get update` works, then install the ROS / system packages we need.
# A failing third-party repo in the base image must not block the build, so
# `apt-get update` is allowed to warn; a missing package still fails the install.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
 && (apt-get update -q || true) \
 && apt-get install -y --no-install-recommends \
        ros-humble-rosbridge-server \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-gazebo-ros2-control \
        ros-humble-ros2-control \
        ros-humble-ros2-controllers \
        ros-humble-cv-bridge \
        ros-humble-xacro \
        ros-humble-robot-state-publisher \
        python3-opencv \
        python3-yaml \
        python3-pytest \
        python3-colcon-common-extensions \
 && rm -rf /var/lib/apt/lists/*

# Python packages. numpy stays below 1.25 (the base image pins it; cv_bridge and
# rclpy are built against the system numpy ABI). CNN inference (pa_perception,
# method:=cnn|hybrid) uses onnxruntime on the container CPU: 1.18.1 is the last
# release built against numpy 1.x (requires numpy>=1.21.6,<2; cp310 wheel).
# onnx is only for building test models / exporting. torch is the CPU wheel for
# training experiments; real training happens on the host GPU.
RUN python3 -m pip install --no-cache-dir "numpy<1.25" "pymunk>=6.6,<8" \
        onnxruntime==1.18.1 onnx==1.16.2 \
 && python3 -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1

# Demo recording (pa_gazebo demo_recorder H.264 re-encode, tools/record_gazebo_screen.sh
# x11grab; xdpyinfo to read the desktop size). Separate layer so the layers above
# stay cached.
RUN (apt-get update -q || true) \
 && apt-get install -y --no-install-recommends ffmpeg x11-utils \
 && rm -rf /var/lib/apt/lists/*

# colcon workspace: the repository is bind-mounted at /ws/src/push-alphabet and
# built from /ws (build/install/log live in named volumes, see compose.yaml).
RUN mkdir -p /ws/src /ws/build /ws/install /ws/log \
 && chown -R ubuntu:ubuntu /ws \
 && { echo ''; \
      echo '# Push-Alphabet'; \
      echo 'source /opt/ros/humble/setup.bash'; \
      echo '[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh'; \
      echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash'; \
    } >> /etc/bash.bashrc

ENV PA_WS=/ws

# Keep the base image's ENTRYPOINT (/entrypoint.sh: VNC + noVNC on port 80).
