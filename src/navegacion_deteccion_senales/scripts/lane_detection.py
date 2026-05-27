import os
import multiprocessing
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime as ort

os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())


def _sharpen_numpy(image):
    """Sharpening con cv2/numpy (backend ONNX, ligero y sin overhead de frameworks)."""
    kernel = np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]], dtype=np.float32)
    sharpened = np.stack(
        [cv2.filter2D(image[:, :, c], -1, kernel) for c in range(3)], axis=-1
    )
    return np.clip(sharpened, 0.0, 1.0)


def _sharpen_image(image):
    """Aplica filtro de sharpening a la imagen usando PyTorch."""
    kernel = torch.tensor([[0., -1., 0.],
                           [-1., 5., -1.],
                           [0., -1., 0.]], dtype=torch.float32)
    # [out_channels, in_channels/groups, kH, kW] -> [3, 1, 3, 3]
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    # image: numpy [H, W, 3] -> tensor [1, 3, H, W]
    x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    sharpened = F.conv2d(x, kernel, groups=3, padding='same')
    sharpened = torch.clamp(sharpened, 0.0, 1.0)
    return sharpened.squeeze(0).permute(1, 2, 0).numpy()


def load_model(model_path="lane_model.onnx"):
    """Carga un modelo ONNX con ONNX Runtime. Retorna InferenceSession."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo ONNX en: {model_path}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)

    input_name = session.get_inputs()[0].name
    raw_shape = session.get_inputs()[0].shape

    # ONNX puede tener dimensiones simbólicas (strings/None); reemplazar por 1
    input_shape = [1 if not isinstance(dim, int) else dim for dim in raw_shape]

    # Warm-up
    dummy = np.zeros(input_shape, dtype=np.float32)
    session.run(None, {input_name: dummy})

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


def mask_to_steering(predicted_mask, threshold=0.5, roi_start=0.5):
    """Convierte una máscara en steering normalizado [-1, 1]."""
    state = mask_to_lane_state(
        predicted_mask,
        original_shape=mask_to_array(predicted_mask).shape,
        threshold=threshold,
        roi_start=roi_start,
    )
    return state['steering']


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
        'right_detected': False,
        'steering': 0.0,
    }


def _clean_binary_mask(mask, threshold):
    binary = (mask > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _row_lane_samples(binary, roi_start):
    h, _ = binary.shape
    y0 = int(h * roi_start)
    samples = []

    for y in range(y0, h):
        active = np.flatnonzero(binary[y])
        if active.size == 0:
            continue
        samples.append((float(active[0]), float(active[-1]), float(y)))

    return samples


def _weighted_lane_center(samples):
    centers = np.array([(left + right) * 0.5 for left, right, _ in samples], dtype=np.float32)
    ys = np.array([y for _, _, y in samples], dtype=np.float32)
    weights = 1.0 + (ys - ys.min()) / max(float(ys.max() - ys.min()), 1.0)
    return float(np.average(centers, weights=weights))


def _fit_line(points, sx, sy):
    if len(points) < 2:
        return None

    pts = np.array(points, dtype=np.float32)
    ys = pts[:, 1]
    y_top = float(np.percentile(ys, 10))
    y_bottom = float(np.percentile(ys, 95))

    if abs(y_bottom - y_top) < 1.0:
        return None

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(vy) < 1e-6:
        return None

    def x_at(y):
        return float(x0 + (y - y0) * vx / vy)

    return [x_at(y_top) * sx, y_top * sy, x_at(y_bottom) * sx, y_bottom * sy]


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
        'right_detected': len(right_points) >= 4,
        'steering': float(np.clip(-(offset_px / center_x), -1.0, 1.0)),
    }


def mask_to_lane_state(predicted_mask, original_shape, threshold=0.5, roi_start=0.5):
    """
    Extrae estado completo del carril desde una máscara de segmentación.

    Parameters
    ----------
    predicted_mask : np.ndarray or tensor
        Máscara de predicción del modelo (H_mask, W_mask) o (H_mask, W_mask, 1).
    original_shape : tuple
        (H_orig, W_orig) del frame original para escalar coordenadas.
    threshold : float
        Umbral de binarización de la máscara.
    roi_start : float
        Fracción de la altura desde donde empieza la región de interés.

    Returns
    -------
    dict
        {'image_width', 'image_height', 'left', 'right', 'offset_px',
         'zone', 'lateral', 'left_detected', 'right_detected', 'steering'}
    """
    mask = mask_to_array(predicted_mask)
    h_mask, w_mask = mask.shape
    h_orig, w_orig = original_shape[:2]

    sx = w_orig / float(w_mask)
    sy = h_orig / float(h_mask)

    binary = _clean_binary_mask(mask, threshold)
    samples = _row_lane_samples(binary, roi_start)

    if len(samples) < 5:
        return _empty_lane_state(w_orig, h_orig)

    left_points = [(left, y) for left, _, y in samples]
    right_points = [(right, y) for _, right, y in samples]
    lane_center_x_mask = _weighted_lane_center(samples)
    image_center_x_mask = w_mask / 2.0
    offset_px_mask = lane_center_x_mask - image_center_x_mask
    offset_px = offset_px_mask * sx

    if abs(offset_px_mask) < w_mask * 0.04:
        zone = 'CENTER'
    elif offset_px_mask < 0:
        zone = 'LEFT'
    else:
        zone = 'RIGHT'

    left = _fit_line(left_points, sx, sy)
    right = _fit_line(right_points, sx, sy)
    lateral = float(np.clip(lane_center_x_mask / w_mask, 0.0, 1.0))
    steering = float(np.clip(-(offset_px_mask / image_center_x_mask), -1.0, 1.0))

    return {
        'image_width': w_orig,
        'image_height': h_orig,
        'left': left,
        'right': right,
        'offset_px': offset_px,
        'zone': zone,
        'lateral': lateral,
        'left_detected': left is not None,
        'right_detected': right is not None,
        'steering': steering,
    }