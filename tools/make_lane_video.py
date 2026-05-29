#!/usr/bin/env python3
"""
make_lane_video.py  —  genera un vídeo .mp4 del detector de carriles
                       sobre las imágenes de un dataset, con overlay de
                       ground truth (máscara binaria) y métrica IoU.

Estructura esperada:
    <dataset_dir>/
        images/   lane_<ts>.jpg
        masks/    lane_<ts>_mask.png   (máscara binaria, opcional)

Uso:
    python3 tools/make_lane_video.py dataset/lanes /path/to/lane_model.onnx
    python3 tools/make_lane_video.py dataset/lanes /path/to/lane_model.onnx \\
        -o output/demo.mp4 --fps 10
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent /
                       "src" / "navegacion_deteccion_senales" / "scripts"))
from lane_detection import LaneDetector


# ── Colores BGR ────────────────────────────────────────────────────────────────
_LEFT_COLOR   = (0,   217, 255)   # amarillo
_RIGHT_COLOR  = (255, 217,   0)   # cian
_POLY_COLOR   = (0,   255, 102)   # verde
_DEV_COLOR    = (0,   255, 255)   # amarillo
_DOT_COLOR    = (0,   255, 255)
_EGO_COLOR    = (255, 255, 255)
_PRED_COLOR   = (255, 100,   0)   # azul   — predicción overlay


def _draw_line(frame, pt1, pt2, color, thickness=4):
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_filled_poly(frame, pts, color, alpha=0.12):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put_text_bg(frame, text, x, y, color, font_scale=0.6, thickness=1):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def _overlay_mask(frame, mask_bin, color, alpha=0.45):
    """Superpone máscara binaria (uint8, 0/255) sobre frame con color y opacidad."""
    overlay = frame.copy()
    overlay[mask_bin > 0] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



def _draw_hud(frame, state: dict, idx: int, total: int):
    h, w = frame.shape[:2]

    left_det  = state.get('left')  is not None
    right_det = state.get('right') is not None

    _put_text_bg(frame, f"L:{int(left_det)}  R:{int(right_det)}",
                 12, 28, (160, 160, 160), 0.6, 1)

    label = f"{idx + 1}/{total}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def render_frame(frame: np.ndarray, detector: LaneDetector,
                 idx: int, total: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    offset, state, pred_mask_f = detector.detect_lane_state(frame)

    # Reconstruir la máscara predicha en el espacio del frame completo.
    # detect_lane_state la devuelve en el espacio del ROI (224×224 sobre la
    # mitad inferior), hay que redimensionarla al ROI y luego insertarla en
    # un canvas vacío del tamaño del frame.
    roi_y0  = int(h * 0.55)
    roi_x0  = int(w * 0.05)
    roi_x1  = int(w * 0.95)
    crop_h  = h - roi_y0
    crop_w  = roi_x1 - roi_x0

    pred_bin_roi = (pred_mask_f > 0.5).astype(np.uint8) * 255
    if pred_bin_roi.ndim == 3:
        pred_bin_roi = pred_bin_roi[:, :, 0]
    pred_bin_roi = cv2.resize(pred_bin_roi, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    pred_bin_roi = (pred_bin_roi > 127).astype(np.uint8) * 255

    pred_bin_rs = np.zeros((h, w), dtype=np.uint8)
    pred_bin_rs[roi_y0:h, roi_x0:roi_x1] = pred_bin_roi

    # Overlay predicción
    _overlay_mask(out, pred_bin_rs, _PRED_COLOR, alpha=0.30)

    left  = state.get('left')
    right = state.get('right')

    # Polígono de carril
    if left and right:
        pts = [
            (int(left[0]),  int(left[1])),
            (int(left[2]),  int(left[3])),
            (int(right[2]), int(right[3])),
            (int(right[0]), int(right[1])),
        ]
        _draw_filled_poly(out, pts, _POLY_COLOR, alpha=0.15)
        cv2.polylines(out, [np.array(pts, dtype=np.int32)],
                      isClosed=True, color=_POLY_COLOR, thickness=1, lineType=cv2.LINE_AA)

    # Líneas de carril
    if left:
        _draw_line(out, (int(left[0]),  int(left[1])),  (int(left[2]),  int(left[3])),  _LEFT_COLOR,  5)
    if right:
        _draw_line(out, (int(right[0]), int(right[1])), (int(right[2]), int(right[3])), _RIGHT_COLOR, 5)

    _draw_hud(out, state, idx, total)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Genera un vídeo .mp4 del detector de carriles sobre el dataset.")
    parser.add_argument("dataset_dir",
                        help="Directorio raíz del dataset (debe contener images/ y opcionalmente masks/)")
    parser.add_argument("model_path",
                        help="Ruta al modelo ONNX de segmentación de carriles")
    parser.add_argument("-o", "--output", default=None,
                        help="Fichero de salida (default: <dataset_dir>/lane_demo.mp4)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames por segundo del vídeo (default: 10)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    images_dir  = dataset_dir / "images"

    if not images_dir.is_dir():
        print(f"ERROR: directorio no encontrado: {images_dir}", file=sys.stderr)
        sys.exit(1)

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        print(f"ERROR: no se encontraron imágenes en {images_dir}", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else dataset_dir / "lane_demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Leer primer frame para determinar resolución
    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"ERROR: no se pudo leer {images[0]}", file=sys.stderr)
        sys.exit(1)
    h, w = first.shape[:2]

    # VideoWriter con fallback - preferir H.264 para compatibilidad WhatsApp
    _candidates = [
        (cv2.VideoWriter_fourcc(*"avc1"), str(output.with_suffix(".mp4"))),   # H.264 (WhatsApp compatible)
        (cv2.VideoWriter_fourcc(*"mp4v"), str(output.with_suffix(".mp4"))),   # MPEG-4 fallback
        (cv2.VideoWriter_fourcc(*"XVID"), str(output.with_suffix(".avi"))),   # Último recurso
    ]
    writer = None
    out_path = None
    for fourcc, out_str in _candidates:
        w_test = cv2.VideoWriter(out_str, fourcc, args.fps, (w, h))
        if w_test.isOpened():
            writer = w_test
            out_path = out_str
            break
        w_test.release()
    if writer is None:
        print("ERROR: no se pudo abrir ningún VideoWriter. "
              "Instala libavcodec o libxvidcore.", file=sys.stderr)
        sys.exit(1)
    output = Path(out_path)

    detector = LaneDetector(model_path=args.model_path)
    total    = len(images)

    print(f"Procesando {total} imágenes → {output}  ({w}x{h} @ {args.fps} fps)")

    for idx, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"  [WARN] no se pudo leer {path.name}, omitiendo")
            continue

        writer.write(render_frame(frame, detector, idx, total))

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  {idx + 1}/{total}")

    writer.release()
    print(f"Vídeo guardado en: {output}")


if __name__ == "__main__":
    main()