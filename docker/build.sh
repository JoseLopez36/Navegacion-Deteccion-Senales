#!/bin/bash
# Build script for multi-stage Docker images
# Usage: ./build.sh [base|extended|all]

set -e

TAG_BASE="navdet-base"
TAG_EXTENDED="navdet-extended"

build_base() {
    echo "=== Building base image: ${TAG_BASE} ==="
    echo "CUDA 12.8 + TensorFlow 2.19"
    docker build -f base.Dockerfile -t ${TAG_BASE}:latest -t ${TAG_BASE}:cuda12.8-tf2.19 .
    echo "✓ Base image built: ${TAG_BASE}"
}

build_extended() {
    echo "=== Building extended image: ${TAG_EXTENDED} ==="
    echo "ROS2 Humble + CARLA 0.9.15 on top of ${TAG_BASE}"
    docker build -f extended.Dockerfile -t ${TAG_EXTENDED}:latest \
        --build-arg BASE_IMAGE=${TAG_BASE}:latest .
    echo "✓ Extended image built: ${TAG_EXTENDED}"
}

case "${1:-all}" in
    base)
        build_base
        ;;
    extended)
        build_extended
        ;;
    all|*)
        build_base
        build_extended
        ;;
esac

echo ""
echo "=== Build complete ==="
echo "Images:"
docker images | grep -E "(navdet-base|navdet-extended|REPOSITORY)" || true