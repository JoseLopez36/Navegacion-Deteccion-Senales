#!/usr/bin/env python3
"""
Nodo para recolección de datos de carriles mediante conducción manual.

Este nodo usa la imagen de segmentación semántica de CARLA para detectar
marcas viales (líneas de carril) con ground truth, guardando las imágenes RGB
junto con una máscara binaria con líneas completas (ground truth de carril).
"""

import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import message_filters

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo


# CARLA semantic segmentation color palette (BGR format)
SEMANTIC_ROAD_LINE = np.array([50, 234, 157])  # BGR – road lines / lane markings
SEMANTIC_TOLERANCE = 10  # Color tolerance for matching


class LaneDatasetNode(Node):
    """
    Nodo de recolección de dataset de carriles usando ground truth de CARLA.

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image (sensor_msgs/Image)
      - /carla/ego_vehicle/rgb_front/camera_info (sensor_msgs/CameraInfo)
      - /carla/ego_vehicle/semantic_segmentation_front/image (sensor_msgs/Image)
      - /carla/ego_vehicle/semantic_segmentation_front/camera_info (sensor_msgs/CameraInfo)

    Funcionalidad:
      - Detecta marcas viales usando segmentación semántica de CARLA (ground truth)
      - Extrae la máscara binaria de marcas viales del ground truth semántico
      - Guarda imagen RGB y máscara binaria de ground truth
    """

    def __init__(self):
        super().__init__('lane_dataset_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('rgb_camera_info_topic', '/carla/ego_vehicle/rgb_front/camera_info')
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/semantic_segmentation_front/image')
        self.declare_parameter('semantic_camera_info_topic', '/carla/ego_vehicle/semantic_segmentation_front/camera_info')
        self.declare_parameter('output_dir', '/home/ros/workspace/src/navegacion_deteccion_senales/lane_dataset')
        self.declare_parameter('capture_rate', 5.0)
        self.declare_parameter('min_lane_area', 100)

        self.image_topic = self.get_parameter('image_topic').value
        self.rgb_camera_info_topic = self.get_parameter('rgb_camera_info_topic').value
        self.semantic_topic = self.get_parameter('semantic_topic').value
        self.semantic_camera_info_topic = self.get_parameter('semantic_camera_info_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.capture_rate = float(self.get_parameter('capture_rate').value)
        self.min_lane_area = int(self.get_parameter('min_lane_area').value)

        # --- Crear directorios de salida ---
        self.images_dir = os.path.join(self.output_dir, 'images')
        self.masks_dir = os.path.join(self.output_dir, 'masks')
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Utilidades ---
        self.bridge = CvBridge()
        self.last_capture_time = 0.0
        self.total_lanes = 0

        # --- Info de cámaras ---
        self.rgb_camera_info = None
        self.semantic_camera_info = None

        # --- Sincronización de imágenes ---
        self.rgb_sub = message_filters.Subscriber(self, Image, self.image_topic, qos_profile=sensor_qos)
        self.semantic_sub = message_filters.Subscriber(self, Image, self.semantic_topic, qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.semantic_sub], queue_size=10, slop=0.03
        )
        self.sync.registerCallback(self._on_images_sync)

        # --- Suscriptores de camera_info ---
        self.create_subscription(CameraInfo, self.rgb_camera_info_topic, self._on_rgb_camera_info, sensor_qos)
        self.create_subscription(CameraInfo, self.semantic_camera_info_topic, self._on_semantic_camera_info, sensor_qos)

        self.get_logger().info('lane_dataset_node iniciado.')
        self.get_logger().info(f'Guardando dataset en: {self.output_dir}')
        self.get_logger().info(f'Capture rate: {self.capture_rate} Hz')

    def _on_rgb_camera_info(self, msg: CameraInfo):
        self.rgb_camera_info = msg

    def _on_semantic_camera_info(self, msg: CameraInfo):
        self.semantic_camera_info = msg

    # ------------------------------------------------------------------
    # Callback sincronizado
    # ------------------------------------------------------------------

    def _on_images_sync(self, rgb_msg: Image, semantic_msg: Image):
        """Callback para imágenes RGB y semántica sincronizadas."""
        current_time = time.time()
        if current_time - self.last_capture_time < 1.0 / self.capture_rate:
            return

        rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        semantic_frame = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='bgr8')

        lane_mask = self._extract_lanes(semantic_frame)

        if lane_mask is not None:
            self.last_capture_time = current_time
            self._save_sample(rgb_frame, lane_mask)

    # ------------------------------------------------------------------
    # Extracción de ground truth de carriles
    # ------------------------------------------------------------------

    def _remap_mask_to_rgb(self, mask_sem):
        """Remap a semantic-space binary mask into RGB camera space using intrinsics."""
        h, w = mask_sem.shape[:2]
        if self.rgb_camera_info is None or self.semantic_camera_info is None:
            return mask_sem.copy()

        rgb_h = self.rgb_camera_info.height
        rgb_w = self.rgb_camera_info.width
        if (rgb_h, rgb_w) == (h, w):
            return mask_sem.copy()

        K_sem = np.array(self.semantic_camera_info.k).reshape(3, 3)
        K_rgb = np.array(self.rgb_camera_info.k).reshape(3, 3)
        uu, vv = np.meshgrid(np.arange(rgb_w, dtype=np.float32),
                             np.arange(rgb_h, dtype=np.float32))
        map_x = ((uu - K_rgb[0, 2]) / K_rgb[0, 0] * K_sem[0, 0] + K_sem[0, 2]).astype(np.float32)
        map_y = ((vv - K_rgb[1, 2]) / K_rgb[1, 1] * K_sem[1, 1] + K_sem[1, 2]).astype(np.float32)
        return cv2.remap(mask_sem, map_x, map_y, cv2.INTER_NEAREST)

    def _extract_lanes(self, semantic_frame):
        """
        Extrae las marcas viales de la imagen de segmentación semántica de CARLA.

        Usa directamente la máscara semántica remapeada al espacio RGB como
        ground truth — las marcas son trazos cortos (dashes) por lo que se
        preservan tal cual, con una dilación leve para engrosarlas.

        Retorna:
          lane_mask (np.ndarray | None): máscara binaria (uint8) en espacio RGB,
                                         o None si no hay marcas visibles.
        """
        mask_sem = cv2.inRange(
            semantic_frame,
            SEMANTIC_ROAD_LINE - SEMANTIC_TOLERANCE,
            SEMANTIC_ROAD_LINE + SEMANTIC_TOLERANCE,
        )
        lane_mask = self._remap_mask_to_rgb(mask_sem)

        if cv2.countNonZero(lane_mask) < self.min_lane_area:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lane_mask = cv2.dilate(lane_mask, kernel, iterations=1)

        return lane_mask

    # ------------------------------------------------------------------
    # Guardado de muestras
    # ------------------------------------------------------------------

    def _save_sample(self, frame, lane_mask):
        """Guarda la imagen RGB y la máscara binaria de ground truth."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_filename = f'lane_{ts}.jpg'
        mask_filename = f'lane_{ts}_mask.png'

        cv2.imwrite(os.path.join(self.images_dir, image_filename), frame)
        cv2.imwrite(os.path.join(self.masks_dir, mask_filename), lane_mask)

        self.total_lanes += 1
        self.get_logger().info(
            f'Guardada muestra: {image_filename} | Total: {self.total_lanes}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LaneDatasetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Recolección finalizada. Marcas de carril detectadas: {node.total_lanes}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()