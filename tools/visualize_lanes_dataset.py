#!/usr/bin/env python3
"""
Script para visualizar el dataset de carriles (RGB + máscara de ground truth).

Uso:
  python3 visualize_lanes_dataset.py <dataset_dir> [options]

Ejemplos:
  # Mostrar todas las imágenes
  python3 visualize_lanes_dataset.py dataset/lanes

  # Guardar imágenes anotadas
  python3 visualize_lanes_dataset.py dataset/lanes --output dataset/lanes/visualized

  # Modo slideshow con delay de 2 segundos
  python3 visualize_lanes_dataset.py dataset/lanes --slideshow --delay 2000
"""

import sys
import argparse
from pathlib import Path

import cv2
import numpy as np


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Superpone la máscara binaria sobre la imagen RGB."""
    overlay = image.copy()
    overlay[mask > 0] = (
        (1 - alpha) * overlay[mask > 0] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return overlay


def process_image(image_path, mask_path, args):
    """Procesa un par imagen/máscara."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: No se pudo cargar {image_path}")
        return False

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Advertencia: No se encontró máscara para {image_path.name}")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

    vis = overlay_mask(image, mask)

    cv2.putText(
        vis, image_path.name, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    if args.output:
        output_path = Path(args.output) / image_path.name
        cv2.imwrite(str(output_path), vis)
        print(f"Guardado: {output_path}")
    else:
        cv2.imshow('Lane Dataset Visualization', vis)
        key = cv2.waitKey(args.delay) if args.slideshow else cv2.waitKey(0)
        if key == ord('q') or key == 27:
            return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Visualiza el dataset de carriles con superposición de máscara',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Teclas:
  q / ESC  - Salir
  ESPACIO  - Siguiente imagen (modo manual)

Ejemplos:
  %(prog)s dataset/lanes
  %(prog)s dataset/lanes --output dataset/lanes/visualized
  %(prog)s dataset/lanes --slideshow --delay 1000
"""
    )
    parser.add_argument('dataset_dir', help='Directorio del dataset de carriles')
    parser.add_argument('--output', help='Directorio para guardar imágenes anotadas')
    parser.add_argument('--slideshow', action='store_true', help='Modo slideshow automático')
    parser.add_argument('--delay', type=int, default=1000, help='Delay en ms entre imágenes (slideshow)')
    parser.add_argument('--limit', type=int, help='Limitar número de imágenes a procesar')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    images_dir = dataset_path / 'images'
    masks_dir = dataset_path / 'masks'

    if not images_dir.exists():
        print(f"Error: No se encontró el directorio de imágenes: {images_dir}")
        sys.exit(1)

    if not masks_dir.exists():
        print(f"Error: No se encontró el directorio de máscaras: {masks_dir}")
        sys.exit(1)

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob('*.jpg')) or sorted(images_dir.glob('*.png'))
    print(f"Encontradas {len(image_files)} imágenes")

    processed = 0
    for i, image_path in enumerate(image_files):
        if args.limit and i >= args.limit:
            break

        mask_path = masks_dir / (image_path.stem + '_mask.png')

        if process_image(image_path, mask_path, args):
            break
        processed += 1

    cv2.destroyAllWindows()
    print(f"\nProcesadas {processed} imágenes")


if __name__ == '__main__':
    main()