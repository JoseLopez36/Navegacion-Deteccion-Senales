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
        self._prev_state: dict | None = None

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

    def detect_lane_state(self, frame: np.ndarray) -> dict:
        """
        Detecta el estado del carril usando el modelo ONNX para segmentación.

        Returns:
            (offset, state, mask)
        """
        orig_h, orig_w = frame.shape[:2]

        # ROI: mitad inferior y franja central
        roi_y0 = int(orig_h * 0.55)
        roi_x0 = int(orig_w * 0.05)
        roi_x1 = int(orig_w * 0.95)
        roi_crop = frame[roi_y0:, roi_x0:roi_x1]

        # Predecir máscara sobre el recorte ROI
        mask = self.predict_lane(roi_crop)

        # Factores de escala: máscara → recorte ROI → imagen original
        crop_h = orig_h - roi_y0
        crop_w = roi_x1 - roi_x0
        h, w = mask.shape[:2]
        scale_x = crop_w / w
        scale_y = crop_h / h

        # Binarizar máscara (aplanar si es 3D)
        if mask.ndim == 3:
            binary = (mask[:, :, 0] > 0.5).astype(np.uint8)
        else:
            binary = (mask > 0.5).astype(np.uint8)

        # Obtener coordenadas de píxeles del carril en espacio de máscara
        ys, xs = np.where(binary > 0)

        if len(xs) == 0:
            return 0.0, self._empty_state(orig_w, orig_h), mask

        # Ajustar líneas en espacio de máscara y reescalar al espacio original
        center_x = w // 2
        y_top    = 0
        y_bottom = int(h * 0.95)

        left_mask  = xs < center_x
        right_mask = xs >= center_x

        left_line  = self._fit_line(list(zip(xs[left_mask],  ys[left_mask])),  y_top, y_bottom) if np.any(left_mask)  else None
        right_line = self._fit_line(list(zip(xs[right_mask], ys[right_mask])), y_top, y_bottom) if np.any(right_mask) else None

        # Reescalar líneas: máscara → imagen original (aplicando offset del recorte)
        def _scale_line(line):
            if line is None:
                return None
            return [line[0] * scale_x + roi_x0, line[1] * scale_y + roi_y0,
                    line[2] * scale_x + roi_x0, line[3] * scale_y + roi_y0]

        left_detected  = _scale_line(left_line)
        right_detected = _scale_line(right_line)

        # Filtrar líneas por inclinación (descartar carriles lejanos demasiado empinados)
        # La pendiente máxima tolerada (dx/dy) para carriles del propio carril
        MAX_LANE_SLOPE = 1.5  # Aproximadamente 56 grados, carriles lejanos son más verticales

        def _is_valid_slope(line):
            if line is None:
                return False
            x_top, y_top, x_bottom, y_bottom = line
            dy = y_bottom - y_top
            if abs(dy) < 1e-6:
                return False
            slope = abs((x_bottom - x_top) / dy)
            return slope <= MAX_LANE_SLOPE

        if not _is_valid_slope(left_detected):
            left_detected = None
        if not _is_valid_slope(right_detected):
            right_detected = None

        # Actualizar _prev_state solo con líneas realmente detectadas en este frame
        if left_detected is not None:
            if self._prev_state is None:
                self._prev_state = {}
            self._prev_state['left'] = left_detected
        if right_detected is not None:
            if self._prev_state is None:
                self._prev_state = {}
            self._prev_state['right'] = right_detected

        # Fallback al estado anterior si falta alguna línea
        left_line  = left_detected  if left_detected  is not None else (self._prev_state or {}).get('left')
        right_line = right_detected if right_detected is not None else (self._prev_state or {}).get('right')

        # Calcular offset lateral usando las líneas detectadas en el punto de control (parte inferior)
        orig_center_x = orig_w / 2.0
        offset = 0.0

        # Extraer posición x de cada línea en el punto de control (índice 2 = x_bottom)
        left_x = left_detected[2] if left_detected is not None else None
        right_x = right_detected[2] if right_detected is not None else None

        # Calcular el offset lateral
        if left_x is not None and right_x is not None:
            # Centro del carril = punto medio entre líneas
            lane_center = (left_x + right_x) / 2.0
            offset = lane_center - orig_center_x

        state = {
            'image_width': orig_w,
            'image_height': orig_h,
            'left': left_line,
            'right': right_line,
            'error': offset
        }
        return offset, state, mask

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
    def _fit_line(points: list, y_top: int, y_bottom: int, max_slope: float = 2.0):
        if len(points) < 4:
            return None
        pts = np.array(points, dtype=np.float32)

        # Create binary image from points for Hough transform
        x_min, x_max = int(pts[:, 0].min()), int(pts[:, 0].max())
        y_min, y_max = int(pts[:, 1].min()), int(pts[:, 1].max())
        w_img = x_max - x_min + 1
        h_img = y_max - y_min + 1

        if w_img < 2 or h_img < 2:
            return None

        binary = np.zeros((h_img, w_img), dtype=np.uint8)
        for x, y in pts:
            binary[int(y - y_min), int(x - x_min)] = 255

        # Hough Transform to detect dominant line
        lines = cv2.HoughLinesP(binary, rho=1, theta=np.pi/180,
                                threshold=10, minLineLength=max(h_img, w_img)//4,
                                maxLineGap=5)

        if lines is None or len(lines) == 0:
            # Fallback to polyfit if Hough fails
            m, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
            if abs(m) > max_slope:
                return None
            return [float(m * y_top + b), float(y_top), float(m * y_bottom + b), float(y_bottom)]

        # Select dominant line (longest)
        best_line = None
        best_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > best_length:
                best_length = length
                best_line = (x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min)

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line
        dx = x2 - x1
        dy = y2 - y1

        if abs(dy) < 1e-6:
            return None

        m = dx / dy
        if abs(m) > max_slope:
            return None

        # Extend line to y_top and y_bottom
        x_top = x1 + m * (y_top - y1)
        x_bottom = x1 + m * (y_bottom - y1)

        return [float(x_top), float(y_top), float(x_bottom), float(y_bottom)]

    @staticmethod
    def _empty_state(width: int, height: int) -> dict:
        return {
            'image_width': width,
            'image_height': height,
            'left': None,
            'right': None
        }