#!/usr/bin/env python3
"""
make_lane_video.py  —  genera un vídeo .mp4 del detector de carriles
                       sobre las imágenes del dataset sintético de CARLA.

Uso:
    python3 tools/make_lane_video.py dataset/lanes/images
    python3 tools/make_lane_video.py dataset/lanes/images -o output/demo.mp4 --fps 10
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent /
                       "src" / "navegacion_deteccion_senales" / "scripts"))
from lane_detection import LaneConfig, LaneDetector


# ── Parámetros por defecto (mismos que params.yaml) ───────────────────────────
DEFAULT_CFG = LaneConfig(
    canny_low        = 50,
    canny_high       = 150,
    hough_rho        = 2,
    hough_threshold  = 50,
    hough_min_len    = 40,
    hough_max_gap    = 100,
    min_slope        = 0.6,
    smoothing        = 8,
    horizon          = 0.5,
    center_threshold = 0.20,
)

# ── Colores BGR ────────────────────────────────────────────────────────────────
_LEFT_COLOR   = (0,   217, 255)   # amarillo
_RIGHT_COLOR  = (255, 217,   0)   # cian
_POLY_COLOR   = (0,   255, 102)   # verde
_DEV_COLOR    = (0,   255, 255)   # amarillo
_DOT_COLOR    = (0,   255, 255)
_EGO_COLOR    = (255, 255, 255)
_ZONE_COLORS  = {
    'CENTER':  (0,   255, 102),
    'LEFT':    (64,   64, 255),
    'RIGHT':   (64,   64, 255),
    'UNKNOWN': (140, 140, 140),
}


def _draw_line(frame, pt1, pt2, color, thickness=4):
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def _draw_filled_poly(frame, pts, color, alpha=0.12):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_hud(frame, state, idx, total):
    h, w = frame.shape[:2]

    zone_color = _ZONE_COLORS.get(state.zone, (255, 255, 255))

    # HUD izquierdo
    cv2.putText(frame, f"Zona: {state.zone}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, zone_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Offset: {state.offset_px:+d} px", (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Lateral: {state.lateral:.2f}  "
                       f"L:{int(state.left_detected)} R:{int(state.right_detected)}",
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # Contador de frame (esquina inferior derecha)
    label = f"{idx + 1}/{total}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def render_frame(frame: np.ndarray, detector: LaneDetector, idx: int, total: int) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cx = w // 2

    left, right = detector._detect(frame, h, w)
    state       = detector._analyze(w, left, right)

    # Polígono de carril
    if left and right:
        pts = [
            (left[0],  left[1]),
            (left[2],  left[3]),
            (right[2], right[3]),
            (right[0], right[1]),
        ]
        _draw_filled_poly(out, pts, _POLY_COLOR, alpha=0.15)
        cv2.polylines(out, [np.array(pts, dtype=np.int32)],
                      isClosed=True, color=_POLY_COLOR, thickness=1, lineType=cv2.LINE_AA)

    # Líneas de carril
    if left:
        _draw_line(out, (left[0],  left[1]),  (left[2],  left[3]),  _LEFT_COLOR,  5)
    if right:
        _draw_line(out, (right[0], right[1]), (right[2], right[3]), _RIGHT_COLOR, 5)

    # Línea de desviación
    offset_px     = float(state.offset_px)
    lane_center_x = int(cx - offset_px)
    y_dev         = int(h * 0.72)

    _draw_line(out, (cx, y_dev), (lane_center_x, y_dev), _DEV_COLOR, 2)
    cv2.circle(out, (lane_center_x, y_dev), 6, _DOT_COLOR,   -1, cv2.LINE_AA)
    cv2.circle(out, (cx,           y_dev), 6, _EGO_COLOR,   -1, cv2.LINE_AA)

    _draw_hud(out, state, idx, total)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Genera un vídeo .mp4 del detector de carriles sobre el dataset.")
    parser.add_argument("images_dir",
                        help="Directorio con las imágenes .jpg del dataset")
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

    output = Path(args.output) if args.output else images_dir.parent / "lane_demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Leer primer frame para determinar resolución
    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"ERROR: no se pudo leer {images[0]}", file=sys.stderr)
        sys.exit(1)
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (w, h))

    detector = LaneDetector(cfg=DEFAULT_CFG)
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