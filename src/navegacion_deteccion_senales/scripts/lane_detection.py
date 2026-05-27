import os
import multiprocessing
import cv2
import numpy as np
import onnxruntime as ort

os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())

_CUDA_OPTIONS = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'cudnn_conv_algo_search': 'HEURISTIC'
}


class LaneDetector:
    """
    Detector de carriles basado en un modelo de segmentación ONNX.

    Uso:
        detector = LaneDetector("/path/to/lane_model.onnx")
        mask = detector.predict_lane(bgr_frame)
        state = detector.detect_lane_state(bgr_frame)
    """

    INPUT_SIZE = (224, 224)

    def __init__(self, model_path: str):
        self._session = self.load_model(model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> ort.InferenceSession:
        """Carga el modelo ONNX con proveedor CUDA (fallback a CPU)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo ONNX en: {model_path}")

        providers = [('CUDAExecutionProvider', _CUDA_OPTIONS), 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)

        self._warmup(model_path, session)
        return session

    def predict_lane(self, frame: np.ndarray) -> np.ndarray:
        """
        Predice la máscara de carril para un frame BGR de OpenCV.

        Returns:
            Máscara de segmentación float32 [H, W] o [H, W, C].
        """
        image = self._preprocess(frame)

        if image.ndim == 3:
            image = image[np.newaxis, ...]  # [1, H, W, C]

        outputs = self._session.run(None, {self._input_name: image})
        pred_mask = outputs[0]

        if pred_mask.ndim == 4:
            pred_mask = pred_mask[0]

        return pred_mask.astype(np.float32)

    def detect_lane_state(self, frame: np.ndarray, roi_start: float = 0.55) -> dict:
        """
        Detecta el estado del carril usando el modelo ONNX para segmentación + Hough.
        Si la máscara del modelo es demasiado escasa, usa HSV como fallback.

        Returns:
            (error_px, state_dict, mask_uint8)
        """
        h, w = frame.shape[:2]
        y_top = int(h * roi_start)
        y_bottom = int(h * 0.92)

        pred = self.predict_lane(frame)
        if pred.ndim == 3:
            pred = pred[:, :, 0]
        mask_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        model_mask = (mask_full > 0.5).astype(np.uint8) * 255

        # Fallback a HSV si el modelo detecta menos del 1 % del ROI
        min_pixels = int(w * h * 0.01)
        if int(np.count_nonzero(model_mask)) < min_pixels:
            mask = self._hsv_mask(frame)
        else:
            mask = model_mask

        roi = np.zeros_like(mask)
        polygon = np.array([[
            (int(w * 0.15), y_bottom),
            (int(w * 0.42), y_top),
            (int(w * 0.58), y_top),
            (int(w * 0.85), y_bottom),
        ]], dtype=np.int32)
        cv2.fillPoly(roi, polygon, 255)
        mask = cv2.bitwise_and(mask, roi)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=25,
            minLineLength=max(25, w // 20),
            maxLineGap=max(20, w // 30),
        )

        if lines is None:
            return 0.0, self._empty_state(w, h), mask

        left_points, right_points = [], []
        center_x = w * 0.5

        for x1, y1, x2, y2 in lines[:, 0].astype(float):
            dx = x2 - x1
            if abs(dx) < 1.0:
                continue
            slope = (y2 - y1) / dx
            if abs(slope) < 0.3:
                continue
            x_mean = (x1 + x2) * 0.5
            if x_mean < center_x:
                left_points.extend([(x1, y1), (x2, y2)])
            else:
                right_points.extend([(x1, y1), (x2, y2)])

        left = self._fit_line(left_points, y_top, y_bottom)
        right = self._fit_line(right_points, y_top, y_bottom)

        if left is None and right is None:
            return 0.0, self._empty_state(w, h), mask

        default_width = w * 0.28
        if left is None:
            left = [right[0] - default_width, right[1], right[2] - default_width, right[3]]
        if right is None:
            right = [left[0] + default_width, left[1], left[2] + default_width, left[3]]

        left_bottom = float(np.clip(left[2], 0, w - 1))
        right_bottom = float(np.clip(right[2], 0, w - 1))
        lane_center = (left_bottom + right_bottom) * 0.5
        error = lane_center - center_x

        if abs(error) < w * 0.04:
            zone = 'CENTER'
        elif error < 0:
            zone = 'LEFT'
        else:
            zone = 'RIGHT'

        return error, {
            'image_width': w,
            'image_height': h,
            'left': [left_bottom, left[3], float(np.clip(left[0], 0, w - 1)), left[1]],
            'right': [right_bottom, right[3], float(np.clip(right[0], 0, w - 1)), right[1]],
            'error': float(error),
            'zone': zone,
            'lateral': float(np.clip(lane_center / w, 0.0, 1.0)),
            'left_detected': len(left_points) >= 4,
            'right_detected': len(right_points) >= 4
        }, mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hsv_mask(frame: np.ndarray) -> np.ndarray:
        """Máscara clásica HSV para marcas blancas y amarillas."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 80, 255]))
        yellow = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([40, 255, 255]))
        return cv2.bitwise_or(white, yellow)

    @staticmethod
    def _preprocess(frame: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
        image = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _warmup(model_path: str, session: ort.InferenceSession) -> None:
        """Warm-up en CPU para evitar OOM si la VRAM está bajo presión al arrancar."""
        try:
            input_name = session.get_inputs()[0].name
            raw_shape = session.get_inputs()[0].shape
            input_shape = [1 if not isinstance(d, int) else d for d in raw_shape]
            warmup_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            warmup_session.run(None, {input_name: np.zeros(input_shape, dtype=np.float32)})
        except Exception:
            pass

    @staticmethod
    def _fit_line(points: list, y_top: int, y_bottom: int):
        if len(points) < 4:
            return None
        pts = np.array(points, dtype=np.float32)
        m, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
        return [float(m * y_top + b), float(y_top), float(m * y_bottom + b), float(y_bottom)]

    @staticmethod
    def _empty_state(width: int, height: int) -> dict:
        return {
            'image_width': width,
            'image_height': height,
            'left': None,
            'right': None,
            'error': 0.0,
            'zone': 'UNKNOWN',
            'lateral': 0.5,
            'left_detected': False,
            'right_detected': False
        }