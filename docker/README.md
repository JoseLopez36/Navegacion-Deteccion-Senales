# Docker Images for Navegación-Detección-Señales

Multi-stage Docker build with CUDA 12.8 and TensorFlow for Blackwell architecture.

## Structure

| Image | Dockerfile | Purpose |
|-------|------------|---------|
| `navdet-base` | `base.Dockerfile` | CUDA 12.8 + cuDNN + TensorFlow 2.19 |
| `navdet-extended` | `extended.Dockerfile` | ROS2 Humble + CARLA 0.9.15 |

## Requirements

- NVIDIA Driver 570+ (required for CUDA 12.8 / Blackwell)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker with BuildKit enabled

## Quick Build

```bash
cd docker

# Build both images
./build.sh

# Or build separately
./build.sh base      # Only base image
./build.sh extended  # Only extended (requires base)
```

## Manual Build

```bash
# Base image (CUDA + TensorFlow)
docker build -f base.Dockerfile -t navdet-base:latest .

# Extended image (ROS2 + CARLA)
docker build -f extended.Dockerfile -t navdet-extended:latest \
    --build-arg BASE_IMAGE=navdet-base:latest .
```

## Usage

### Base Image (ML/DL only)

```bash
docker run --gpus all -it navdet-base:latest
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Extended Image (Full stack)

```bash
docker run --gpus all -it \
    -e CARLA_HOST=localhost \
    -e CARLA_PORT=2001 \
    -v $(pwd)/../:/home/ros/workspace/src/navdet \
    navdet-extended:latest
```

## TensorFlow GPU Verification

```python
import tensorflow as tf
print("TF version:", tf.__version__)
print("CUDA:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN:", tf.sysconfig.get_build_info()["cudnn_version"])
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

## Notes

- Blackwell requires CUDA 12.8+ and TensorFlow 2.19+
- Base image uses `tensorflow[and-cuda]` for bundled CUDA libraries
- ROS2 Humble is based on Ubuntu 22.04 (matches CUDA image)
- CARLA 0.9.15 client with ros-bridge pre-built