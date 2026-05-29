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
import bisect
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
_CNN_CLASS_NAMES = {0: "speed_limit_30", 1: "speed_limit_60", 2: "speed_limit_90", 3: "stop"}

_COLOR_MATCH   = (0,   200,  60)   # verde  — predicción correcta
_COLOR_MISMATCH = (0,   40,  220)  # rojo   — predicción incorrecta
_COLOR_GT      = (0,   200,  60)   # verde  — etiqueta GT
_COLOR_NO_GT   = (120, 120, 120)   # gris   — sin GT disponible
_DEFAULT_COLOR = (0,   220, 220)   # cian   — fallback
_BBOX_PADDING  = 20                # px de margen añadido a la bbox


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


def _put_label_bg(frame: np.ndarray, text: str, x: int, y: int,
                  color: tuple, font_scale: float = 0.5, thickness: int = 1) -> int:
    """Dibuja texto con fondo opaco. Devuelve la altura ocupada."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return th + baseline + 6


def _draw_detection(frame: np.ndarray, x: int, y: int, w: int, h: int,
                    pred_label: str, confidence: float, gt_label: str | None) -> None:
    has_gt = gt_label is not None
    match  = has_gt and (pred_label == gt_label)

    box_color = _COLOR_MATCH if match else _COLOR_MISMATCH
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2, cv2.LINE_AA)

    label_y = y - 6 if y - 6 > 28 else y + h + 22

    pred_text = f"PRED: {pred_label} ({confidence:.2f})"
    row_h = _put_label_bg(frame, pred_text, x, label_y, box_color)

    if has_gt:
        gt_text = f"GT:   {gt_label}"
        _put_label_bg(frame, gt_text, x, label_y + row_h, _COLOR_GT)


def _draw_hud(frame: np.ndarray, n_detections: int, idx: int, total: int) -> None:
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Senales: {n_detections}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    label = f"{idx + 1}/{total}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def _stem_to_us(stem: str) -> int:
    """
    Extrae un timestamp en microsegundos desde un stem con formato
    sign_YYYYMMDD_HHMMSS_uuuuuu  o  crop_YYYYMMDD_HHMMSS_uuuuuu_idx.
    Devuelve -1 si no es posible parsearlo.
    """
    parts = stem.split("_")
    try:
        # partes: [prefix, YYYYMMDD, HHMMSS, uuuuuu, ...]
        date_s  = parts[1]   # YYYYMMDD
        time_s  = parts[2]   # HHMMSS
        us_s    = parts[3]   # uuuuuu
        return int(date_s + time_s + us_s)
    except (IndexError, ValueError):
        return -1


def _load_gt_index(dataset_dir: Path) -> dict[str, list[str | None]]:
    """
    Lee todas las anotaciones de classification/annotations/ y construye un
    índice {stem_detección: [gt_label_det0, gt_label_det1, ...]}.

    Los timestamps entre detection e imágenes de clasificación pueden diferir
    ligeramente; se busca el stem de detección más cercano en tiempo.
    """
    annots_dir  = dataset_dir / "classification" / "annotations"
    det_img_dir = dataset_dir / "detection" / "images"

    # Construir lista ordenada de (timestamp_us, stem) de detección
    det_stems_ts = sorted(
        (_stem_to_us(p.stem), p.stem)
        for p in det_img_dir.glob("sign_*.jpg")
        if _stem_to_us(p.stem) >= 0
    )
    det_ts_arr = [t for t, _ in det_stems_ts]

    def _nearest_det_stem(ts: int) -> str | None:
        if not det_stems_ts or ts < 0:
            return None
        i = bisect.bisect_left(det_ts_arr, ts)
        candidates = []
        if i < len(det_stems_ts):
            candidates.append(det_stems_ts[i])
        if i > 0:
            candidates.append(det_stems_ts[i - 1])
        return min(candidates, key=lambda x: abs(x[0] - ts))[1]

    index: dict[str, dict[int, str]] = {}
    if annots_dir.is_dir():
        for json_path in sorted(annots_dir.glob("*.json")):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            gt_class  = data.get("class", None)
            crop_stem = Path(data.get("image", "")).stem   # crop_<ts>_<idx>
            crop_ts   = _stem_to_us(crop_stem)
            det_stem  = _nearest_det_stem(crop_ts)
            if det_stem is None:
                continue
            try:
                det_idx = int(crop_stem.rsplit("_", 1)[-1])
            except ValueError:
                det_idx = 0
            index.setdefault(det_stem, {})[det_idx] = gt_class
    # Convertir a listas ordenadas por índice de detección
    return {stem: [d[i] for i in sorted(d)] for stem, d in index.items()}


def render_frame(frame: np.ndarray, detections: list,
                 model: CNN, device: torch.device,
                 idx: int, total: int,
                 gt_labels: list[str | None] | None = None) -> np.ndarray:
    out = frame.copy()
    fh, fw = out.shape[:2]

    for det_idx, det in enumerate(detections):
        bx, by, bw, bh = det['bbox']
        gt_label = (gt_labels[det_idx]
                    if gt_labels and det_idx < len(gt_labels)
                    else None)

        # Crop con padding para la inferencia
        pad = _BBOX_PADDING
        cx1 = max(0, bx - pad);       cy1 = max(0, by - pad)
        cx2 = min(fw, bx + bw + pad); cy2 = min(fh, by + bh + pad)

        if cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2]
            pred_label, conf = _classify_crop(model, crop, device)
        else:
            pred_label, conf = "traffic_sign", 1.0

        # Bbox dibujada con padding
        dx1 = max(0, bx - pad);       dy1 = max(0, by - pad)
        dx2 = min(fw, bx + bw + pad); dy2 = min(fh, by + bh + pad)
        _draw_detection(out, dx1, dy1, dx2 - dx1, dy2 - dy1, pred_label, conf, gt_label)

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

    dataset_dir   = Path(args.dataset_dir)
    images_dir    = dataset_dir / "detection" / "images"
    annots_dir    = dataset_dir / "detection" / "annotations"
    gt_index      = _load_gt_index(dataset_dir)
    print(f"GT index: {len(gt_index)} imágenes con anotaciones de clasificación.")

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
        gt_labels  = gt_index.get(img_path.stem, None)

        writer.write(render_frame(frame, detections, model, device, idx, total, gt_labels))

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  {idx + 1}/{total}")

    writer.release()
    print(f"Vídeo guardado en: {output}")


if __name__ == "__main__":
    main()