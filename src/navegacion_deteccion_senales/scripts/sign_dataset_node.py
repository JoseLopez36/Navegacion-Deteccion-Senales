#!/usr/bin/env python3
"""
Nodo para recolección de datos de señales de tráfico mediante conducción manual.

Este nodo usa la imagen de segmentación semántica de CARLA para detectar
señales de tráfico con ground truth y genera dos datasets:

  1. Detection dataset  — imagen RGB completa + anotaciones JSON con bounding boxes.
       output_dir/detection/images/<stem>.jpg
       output_dir/detection/annotations/<stem>.json

  2. Classification dataset — recorte de la señal con margen para etiquetado manual.
       output_dir/classification/images/<stem>_<idx>.jpg
"""

import os
import json
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
SEMANTIC_TRAFFIC_SIGN = np.array([0, 220, 220])  # BGR for yellow
SEMANTIC_TOLERANCE = 10  # Color tolerance for matching


class SignDatasetNode(Node):
    """
    Nodo de recolección de dataset de señales de tráfico usando ground truth de CARLA.

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image (sensor_msgs/Image)
      - /carla/ego_vehicle/rgb_front/camera_info (sensor_msgs/CameraInfo)
      - /carla/ego_vehicle/semantic_segmentation_front/image (sensor_msgs/Image)
      - /carla/ego_vehicle/semantic_segmentation_front/camera_info (sensor_msgs/CameraInfo)

    Funcionalidad:
      - Detecta señales usando segmentación semántica de CARLA (ground truth)
      - Extrae bounding boxes de las señales detectadas
      - Dataset de detección: imagen RGB completa + JSON con bounding boxes
      - Dataset de clasificación: recortes de señal con margen para etiquetado manual
    """

    def __init__(self):
        super().__init__('sign_dataset_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('rgb_camera_info_topic', '/carla/ego_vehicle/rgb_front/camera_info')
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/semantic_segmentation_front/image')
        self.declare_parameter('semantic_camera_info_topic', '/carla/ego_vehicle/semantic_segmentation_front/camera_info')
        self.declare_parameter('output_dir', '/home/ros/workspace/src/navegacion_deteccion_senales/dataset')
        self.declare_parameter('capture_rate', 5.0)
        self.declare_parameter('min_sign_area', 200)
        self.declare_parameter('max_sign_area', 50000)
        self.declare_parameter('crop_margin', 10)

        self.image_topic = self.get_parameter('image_topic').value
        self.rgb_camera_info_topic = self.get_parameter('rgb_camera_info_topic').value
        self.semantic_topic = self.get_parameter('semantic_topic').value
        self.semantic_camera_info_topic = self.get_parameter('semantic_camera_info_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.capture_rate = float(self.get_parameter('capture_rate').value)
        self.min_sign_area = int(self.get_parameter('min_sign_area').value)
        self.max_sign_area = int(self.get_parameter('max_sign_area').value)
        self.crop_margin = int(self.get_parameter('crop_margin').value)

        # --- Crear directorios de salida ---
        # Detection dataset
        self.det_images_dir = os.path.join(self.output_dir, 'detection', 'images')
        self.det_annotations_dir = os.path.join(self.output_dir, 'detection', 'annotations')
        os.makedirs(self.det_images_dir, exist_ok=True)
        os.makedirs(self.det_annotations_dir, exist_ok=True)

        # Classification dataset
        self.cls_images_dir = os.path.join(self.output_dir, 'classification', 'images')
        self.cls_annotations_dir = os.path.join(self.output_dir, 'classification', 'annotations')
        os.makedirs(self.cls_images_dir, exist_ok=True)
        os.makedirs(self.cls_annotations_dir, exist_ok=True)

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
        self.total_signs = 0

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

        self.get_logger().info('sign_dataset_node iniciado.')
        self.get_logger().info(f'Guardando dataset en: {self.output_dir}')
        self.get_logger().info(f'  Detección:      {self.output_dir}/detection/')
        self.get_logger().info(f'  Clasificación:  {self.output_dir}/classification/')
        self.get_logger().info(f'Capture rate: {self.capture_rate} Hz | Crop margin: {self.crop_margin} px')

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

        detections = self._detect_signs(semantic_frame)

        if detections:
            self.last_capture_time = current_time
            self._save_detection_sample(rgb_frame, detections, rgb_msg.header.stamp)
            self._save_classification_crops(rgb_frame, detections)

    # ------------------------------------------------------------------
    # Detección de señales desde segmentación semántica
    # ------------------------------------------------------------------

    def _map_bbox_to_rgb(self, x, y, w, h, semantic_frame_shape):
        """Map bounding box from semantic image to RGB image using camera intrinsics."""
        if self.rgb_camera_info is None or self.semantic_camera_info is None:
            return x, y, w, h

        K_sem = np.array(self.semantic_camera_info.k).reshape(3, 3)
        K_rgb = np.array(self.rgb_camera_info.k).reshape(3, 3)
        rgb_h, rgb_w = self.rgb_camera_info.height, self.rgb_camera_info.width

        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
        norm_x = (corners[:, 0] - K_sem[0, 2]) / K_sem[0, 0]
        norm_y = (corners[:, 1] - K_sem[1, 2]) / K_sem[1, 1]
        proj_x = np.clip(K_rgb[0, 0] * norm_x + K_rgb[0, 2], 0, rgb_w - 1)
        proj_y = np.clip(K_rgb[1, 1] * norm_y + K_rgb[1, 2], 0, rgb_h - 1)

        x_new, y_new = int(proj_x.min()), int(proj_y.min())
        return x_new, y_new, int(proj_x.max()) - x_new, int(proj_y.max()) - y_new

    def _detect_signs(self, semantic_frame):
        """Detecta señales de tráfico en la imagen de segmentación semántica."""
        mask = cv2.inRange(
            semantic_frame,
            SEMANTIC_TRAFFIC_SIGN - SEMANTIC_TOLERANCE,
            SEMANTIC_TRAFFIC_SIGN + SEMANTIC_TOLERANCE,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_sign_area < area < self.max_sign_area:
                x, y, w, h = self._map_bbox_to_rgb(*cv2.boundingRect(cnt), semantic_frame.shape)
                detections.append({'bbox': [x, y, w, h], 'class': 'traffic_sign', 'confidence': 1.0})

        return self._nms(detections)

    def _nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression simple."""
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            x1, y1, w1, h1 = best['bbox']
            detections = [
                d for d in detections
                if self._iou((x1, y1, w1, h1), d['bbox']) < iou_threshold
            ]
        return keep

    def _iou(self, box1, box2):
        """Intersection over Union de dos bounding boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0

    # ------------------------------------------------------------------
    # Guardado de muestras
    # ------------------------------------------------------------------

    def _save_detection_sample(self, frame, detections, timestamp):
        """Dataset de detección: guarda la imagen RGB completa y anotaciones JSON con bounding boxes."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_filename = f'sign_{ts}.jpg'
        json_filename = f'sign_{ts}.json'

        cv2.imwrite(os.path.join(self.det_images_dir, image_filename), frame)

        annotation = {
            'image': f'detection/images/{image_filename}',
            'timestamp': timestamp.sec + timestamp.nanosec * 1e-9,
            'image_size': {
                'width': frame.shape[1],
                'height': frame.shape[0],
                'channels': frame.shape[2],
            },
            'detections': detections,
            'num_signs': len(detections),
        }

        with open(os.path.join(self.det_annotations_dir, json_filename), 'w') as f:
            json.dump(annotation, f, indent=2)

        self.total_signs += len(detections)
        self.get_logger().info(
            f'[Detección]      {image_filename} | '
            f'Señales: {len(detections)} | '
            f'Total: {self.total_signs}'
        )

    def _save_classification_crops(self, frame, detections):
        """Dataset de clasificación: guarda un recorte con margen por cada señal detectada."""
        img_h, img_w = frame.shape[:2]
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        saved = 0
        for idx, det in enumerate(detections):
            x, y, w, h = det['bbox']

            if x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h:
                self.get_logger().debug(
                    f'[Clasificación]  Señal parcial descartada (bbox toca el borde): '
                    f'x={x} y={y} w={w} h={h} img={img_w}x{img_h}'
                )
                continue

            x1 = max(0, x - self.crop_margin)
            y1 = max(0, y - self.crop_margin)
            x2 = min(img_w, x + w + self.crop_margin)
            y2 = min(img_h, y + h + self.crop_margin)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_stem = f'crop_{ts}_{idx}'
            cv2.imwrite(os.path.join(self.cls_images_dir, f'{crop_stem}.jpg'), crop)

            annotation = {
                'image': f'classification/images/{crop_stem}.jpg',
                'class': 'unknown',
                'source_image': f'detection/images/sign_{ts}.jpg',
                'bbox': {'x': x, 'y': y, 'w': w, 'h': h},
                'crop_size': {'width': x2 - x1, 'height': y2 - y1},
            }
            with open(os.path.join(self.cls_annotations_dir, f'{crop_stem}.json'), 'w') as f:
                json.dump(annotation, f, indent=2)
            saved += 1

        if saved:
            self.get_logger().info(
                f'[Clasificación]  {saved} recorte(s) guardado(s) en {self.cls_images_dir}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SignDatasetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'Recolección finalizada. Señales detectadas: {node.total_signs}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()