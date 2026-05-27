import os
import multiprocessing
import cv2
import numpy as np
import onnxruntime as ort

os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())


def _sharpen_numpy(image):
    """Sharpening con cv2/numpy (backend ONNX, ligero y sin overhead de frameworks)."""
    kernel = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]], dtype=np.float32)
    sharpened = np.stack(
        [cv2.filter2D(image[:, :, c], -1, kernel) for c in range(3)], axis=-1
    )
    return np.clip(sharpened, 0.0, 1.0)


def load_model(model_path="lane_model.onnx"):
    """Carga un modelo ONNX con ONNX Runtime. Retorna InferenceSession."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo ONNX en: {model_path}")

    cuda_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2 GB
        'cudnn_conv_algo_search': 'HEURISTIC',
    }
    providers = [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    raw_shape = session.get_inputs()[0].shape

    # ONNX puede tener dimensiones simbólicas (strings/None); reemplazar por 1
    input_shape = [1 if not isinstance(dim, int) else dim for dim in raw_shape]

    # Warm-up en CPU
    try:
        warmup_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        dummy = np.zeros(input_shape, dtype=np.float32)
        warmup_session.run(None, {input_name: dummy})
        del warmup_session
    except Exception:
        pass

    return session


def preprocess_frame(frame, size=(224, 224)):
    """
    Preprocesa un frame de OpenCV (numpy array) para el modelo.
    Retorna numpy float32 [H, W, 3]; el sharpening y batching se aplican en cada backend.
    """
    image = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def mask_to_array(predicted_mask):
    """Convierte máscara de predicción a array numpy 2D float32."""
    mask = np.array(predicted_mask)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.float32)


def predict_lane(image, model):
    """Predice la máscara de carril con ONNX Runtime."""
    session = model  # model es una ort.InferenceSession

    sharpened = _sharpen_numpy(image)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # Asegurar batch dim si el modelo espera 4D [N, H, W, C]
    if len(input_shape) == 4 and sharpened.ndim == 3:
        sharpened = np.expand_dims(sharpened, axis=0)

    outputs = session.run(None, {input_name: sharpened})
    pred_mask = outputs[0]

    # Si la salida tiene batch dim, quitarla
    if pred_mask.ndim == 4:
        pred_mask = pred_mask[0]

    return pred_mask.astype(np.float32)

def _empty_lane_state(width, height):
    return {
        'image_width': width,
        'image_height': height,
        'left': None,
        'right': None,
        'offset_px': 0.0,
        'zone': 'UNKNOWN',
        'lateral': 0.5,
        'left_detected': False,
        'right_detected': False
    }


def _fit_x_from_y(points, y_top, y_bottom):
    if len(points) < 4:
        return None

    pts = np.array(points, dtype=np.float32)
    m, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    x_top = float(m * y_top + b)
    x_bottom = float(m * y_bottom + b)
    return [x_top, float(y_top), x_bottom, float(y_bottom)]


def detect_lane_state(frame, roi_start=0.55):
    h, w = frame.shape[:2]
    y_top = int(h * roi_start)
    y_bottom = int(h * 0.92)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 80, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 60, 80]), np.array([40, 255, 255]))
    mask = cv2.bitwise_or(white, yellow)

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
        return _empty_lane_state(w, h)

    left_points = []
    right_points = []
    center_x = w * 0.5

    for line in lines[:, 0]:
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1.0:
            continue
        slope = dy / dx
        if abs(slope) < 0.35:
            continue

        x_mean = (x1 + x2) * 0.5
        if slope < 0 and x_mean < center_x:
            left_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and x_mean > center_x:
            right_points.extend([(x1, y1), (x2, y2)])

    left = _fit_x_from_y(left_points, y_top, y_bottom)
    right = _fit_x_from_y(right_points, y_top, y_bottom)
    default_lane_width = w * 0.28

    if left is None and right is None:
        return _empty_lane_state(w, h)
    if left is None:
        left = [right[0] - default_lane_width, right[1], right[2] - default_lane_width, right[3]]
    if right is None:
        right = [left[0] + default_lane_width, left[1], left[2] + default_lane_width, left[3]]

    left_draw_bottom = float(np.clip(left[2], 0, w - 1))
    right_draw_bottom = float(np.clip(right[2], 0, w - 1))
    left_bottom = left_draw_bottom
    right_bottom = right_draw_bottom
    lane_center = (left_bottom + right_bottom) * 0.5
    offset_px = lane_center - center_x

    if abs(offset_px) < w * 0.04:
        zone = 'CENTER'
    elif offset_px < 0:
        zone = 'LEFT'
    else:
        zone = 'RIGHT'

    return {
        'image_width': w,
        'image_height': h,
        'left': [left_draw_bottom, left[3], float(np.clip(left[0], 0, w - 1)), left[1]],
        'right': [right_draw_bottom, right[3], float(np.clip(right[0], 0, w - 1)), right[1]],
        'offset_px': float(offset_px),
        'zone': zone,
        'lateral': float(np.clip(lane_center / w, 0.0, 1.0)),
        'left_detected': len(left_points) >= 4,
        'right_detected': len(right_points) >= 4
    }