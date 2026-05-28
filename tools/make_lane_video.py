#!/usr/bin/env python3
"""
make_lane_video.py  —  genera un vídeo .mp4 del detector de carriles
                       sobre las imágenes de un dataset.

Uso:
    python3 tools/make_lane_video.py dataset/lanes/images /path/to/lane_model.onnx
    python3 tools/make_lane_video.py dataset/lanes/images /path/to/lane_model.onnx \\
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


def _draw_line(frame, pt1, pt2, color, thickness=4):
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_filled_poly(frame, pts, color, alpha=0.12):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_hud(frame, state: dict, offset: float, idx: int, total: int):
    h, w = frame.shape[:2]

    left_det  = state.get('left')  is not None
    right_det = state.get('right') is not None

    # HUD izquierdo
    cv2.putText(frame, f"Offset: {offset:+.1f} px", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"L:{int(left_det)}  R:{int(right_det)}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    # Contador de frame (esquina inferior derecha)
    label = f"{idx + 1}/{total}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def render_frame(frame: np.ndarray, detector: LaneDetector, idx: int, total: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cx = w // 2

    offset, state, _mask = detector.detect_lane_state(frame)

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

    # Línea de desviación
    lane_center_x = int(cx + offset)
    y_dev         = int(h * 0.72)

    _draw_line(out, (cx, y_dev), (lane_center_x, y_dev), _DEV_COLOR, 2)
    cv2.circle(out, (lane_center_x, y_dev), 6, _DOT_COLOR, -1, cv2.LINE_AA)
    cv2.circle(out, (cx,           y_dev), 6, _EGO_COLOR,  -1, cv2.LINE_AA)

    _draw_hud(out, state, offset, idx, total)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Genera un vídeo .mp4 del detector de carriles sobre el dataset.")
    parser.add_argument("images_dir",
                        help="Directorio con las imágenes .jpg/.png del dataset")
    parser.add_argument("model_path",
                        help="Ruta al modelo ONNX de segmentación de carriles")
    parser.add_argument("-o", "--output", default=None,
                        help="Fichero de salida (default: <images_dir>/../lane_demo.mp4)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames por segundo del vídeo (default: 10)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"ERROR: directorio no encontrado: {images_dir}", file=sys.stderr)
        sys.exit(1)

    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        print(f"ERROR: no se encontraron imágenes en {images_dir}", file=sys.stderr)
        sys.exit(1)

    output = Path(args.output) if args.output else images_dir.parent / "lane_demo.avi"
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