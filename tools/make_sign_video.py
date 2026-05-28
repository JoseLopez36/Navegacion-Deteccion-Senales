#!/usr/bin/env python3
"""
make_sign_video.py  —  genera un vídeo .mp4 con clasificación de señales de tráfico
                       usando los bounding boxes del dataset de detección como
                       ground truth y la CNN de sign_classification.py.

Estructura esperada (generada por sign_dataset_node.py):
    <dataset_dir>/
        detection/
            images/       sign_<ts>.jpg
            annotations/  sign_<ts>.json
        classification/
            images/       crop_<ts>_<idx>.jpg

El JSON de detección tiene el formato:
    {"detections": [{"bbox": [x, y, w, h], "class": "traffic_sign", "confidence": 1.0}]}

Uso:
    python3 tools/make_sign_video.py dataset/signs /path/to/weights.pth
    python3 tools/make_sign_video.py dataset/signs /path/to/weights.pth \\
        -o output/signs_demo.avi --fps 10
"""

import argparse
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent /
                       "src" / "navegacion_deteccion_senales" / "scripts"))
from sign_classification import CNN, process_image


# ── Clases de salida del clasificador CNN ─────────────────────────────────────
_CNN_CLASS_NAMES = {0: "speed_limit_30", 1: "speed_limit_60", 2: "speed_limit_90"}

# ── Colores BGR por clase CNN ──────────────────────────────────────────────────
_CLASS_COLORS = {
    "speed_limit_30": (200, 0,   200),
    "speed_limit_60": (180, 0,   220),
    "speed_limit_90": (160, 0,   240),
}
_DEFAULT_COLOR = (0, 220, 220)   # cian amarillento — color semántico de CARLA


def _load_model(weights_path: str, device: torch.device) -> CNN:
    model = CNN()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)


def _classify_crop(model: CNN, crop: np.ndarray, device: torch.device) -> tuple[str, float]:
    """Clasifica un recorte BGR y devuelve (nombre_clase, confianza)."""
    tensor = process_image(crop).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = int(probs.argmax().item())
    return _CNN_CLASS_NAMES[idx], float(probs[idx].item())


def _draw_detection(frame: np.ndarray, x: int, y: int, w: int, h: int,
                    label: str, confidence: float) -> None:
    color = _CLASS_COLORS.get(label, _DEFAULT_COLOR)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)

    text = f"{label}: {confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_y = y - 10 if y - 10 > th + 4 else y + h + th + 10

    cv2.rectangle(frame,
                  (x, label_y - th - 4),
                  (x + tw, label_y + baseline),
                  color, -1)
    cv2.putText(frame, text, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, n_detections: int, idx: int, total: int) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Senales: {n_detections}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    label = f"{idx + 1}/{total}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def render_frame(frame: np.ndarray, detections: list,
                 model: CNN, device: torch.device,
                 idx: int, total: int) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]

    for det in detections:
        bx, by, bw, bh = det['bbox']

        x1 = max(0, bx);        y1 = max(0, by)
        x2 = min(fw, bx + bw);  y2 = min(fh, by + bh)

        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            label, conf = _classify_crop(model, crop, device)
        else:
            label, conf = "traffic_sign", 1.0

        _draw_detection(out, bx, by, bw, bh, label, conf)

    _draw_hud(out, len(detections), idx, total)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Genera un vídeo con clasificación de señales sobre el dataset.")
    parser.add_argument("dataset_dir",
                        help="Directorio raíz del dataset (debe contener detection/images/ y detection/annotations/)")
    parser.add_argument("weights_path",
                        help="Ruta al fichero de pesos del modelo CNN (.pth)")
    parser.add_argument("-o", "--output", default=None,
                        help="Fichero de salida (default: <dataset_dir>/signs_demo.mp4)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames por segundo del vídeo (default: 10)")
    parser.add_argument("--cpu", action="store_true",
                        help="Forzar uso de CPU aunque CUDA esté disponible")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_dir  = dataset_dir / "detection" / "images"
    annots_dir  = dataset_dir / "detection" / "annotations"

    for d in (images_dir, annots_dir):
        if not d.is_dir():
            print(f"ERROR: directorio no encontrado: {d}", file=sys.stderr)
            sys.exit(1)

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        print(f"ERROR: no se encontraron imágenes en {images_dir}", file=sys.stderr)
        sys.exit(1)

    # Solo procesar imágenes que tengan anotación
    pairs = []
    for img_path in images:
        ann_path = annots_dir / (img_path.stem + ".json")
        if ann_path.exists():
            pairs.append((img_path, ann_path))
        else:
            print(f"  [WARN] sin anotación: {img_path.name}, omitiendo")

    if not pairs:
        print("ERROR: no se encontraron pares imagen/anotación.", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else dataset_dir / "detection" / "signs_demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Resolución desde primer frame
    first = cv2.imread(str(pairs[0][0]))
    if first is None:
        print(f"ERROR: no se pudo leer {pairs[0][0]}", file=sys.stderr)
        sys.exit(1)
    fh, fw = first.shape[:2]

    # VideoWriter con fallback - preferir H.264 para compatibilidad WhatsApp
    _candidates = [
        (cv2.VideoWriter_fourcc(*"avc1"), str(output.with_suffix(".mp4"))),   # H.264
        (cv2.VideoWriter_fourcc(*"mp4v"), str(output.with_suffix(".mp4"))),   # MPEG-4 fallback
        (cv2.VideoWriter_fourcc(*"XVID"), str(output.with_suffix(".avi"))),   # Último recurso
    ]
    writer   = None
    out_path = None
    for fourcc, out_str in _candidates:
        w_test = cv2.VideoWriter(out_str, fourcc, args.fps, (fw, fh))
        if w_test.isOpened():
            writer   = w_test
            out_path = out_str
            break
        w_test.release()
    if writer is None:
        print("ERROR: no se pudo abrir ningún VideoWriter. "
              "Instala libxvidcore o libavcodec.", file=sys.stderr)
        sys.exit(1)
    output = Path(out_path)

    # Dispositivo
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Dispositivo: {device}")

    model = _load_model(args.weights_path, device)
    total = len(pairs)
    print(f"Procesando {total} imágenes → {output}  ({fw}x{fh} @ {args.fps} fps)")

    for idx, (img_path, ann_path) in enumerate(pairs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  [WARN] no se pudo leer {img_path.name}, omitiendo")
            continue

        with open(ann_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
        detections = annotation.get('detections', [])

        writer.write(render_frame(frame, detections, model, device, idx, total))

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  {idx + 1}/{total}")

    writer.release()
    print(f"Vídeo guardado en: {output}")


if __name__ == "__main__":
    main()