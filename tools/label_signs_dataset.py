#!/usr/bin/env python3
"""
Etiquetado manual interactivo del dataset de clasificación de señales.

Muestra cada recorte de señal y pide al usuario que introduzca su clase.
El resultado se guarda en el JSON de anotación correspondiente.

Uso:
  python3 label_signs_dataset.py <dataset_dir>

Ejemplo:
  python3 label_signs_dataset.py ../dataset/signs

Teclas durante la revisión:
  Escribir clase + Enter  — asigna la clase y avanza
  Enter (vacío)           — mantiene la clase actual y avanza
  s + Enter               — salta sin guardar
  q + Enter               — guarda y sale
"""

import sys
import json
import argparse
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def label_dataset(dataset_dir: Path, relabel: bool = False):
    annotations_dir = dataset_dir / 'classification' / 'annotations'
    images_dir = dataset_dir / 'classification' / 'images'

    if not annotations_dir.exists():
        print(f'Error: no se encontró {annotations_dir}')
        sys.exit(1)

    annotation_files = sorted(annotations_dir.glob('*.json'))
    total = len(annotation_files)
    print(f'Encontradas {total} anotaciones.\n')
    print('Clases sugeridas: speed_limit_30, speed_limit_60, speed_limit_90, stop, give_way, ...')
    print('Enter vacío = mantener clase actual | s = saltar | q = guardar y salir\n')

    labeled = 0
    skipped = 0

    for i, ann_path in enumerate(annotation_files):
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        current_class = annotation.get('class', 'unknown')

        if current_class != 'unknown' and not relabel:
            skipped += 1
            continue

        crop_image_rel = annotation.get('image', '')
        crop_path = (dataset_dir / crop_image_rel) if crop_image_rel else (images_dir / ann_path.stem).with_suffix('.jpg')

        crop = cv2.imread(str(crop_path))
        if crop is None:
            print(f'[{i+1}/{total}] ERROR: no se pudo cargar {crop_path}')
            skipped += 1
            continue

        display = cv2.resize(crop, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Senal', display)
        cv2.waitKey(500)

        prompt = f'[{i+1}/{total}] {ann_path.stem} (actual: {current_class}) > '
        user_input = input(prompt).strip()
        cv2.waitKey(1)

        if user_input == 'q':
            print('Saliendo...')
            break
        elif user_input == 's':
            skipped += 1
            continue
        elif user_input == '':
            skipped += 1
            continue
        else:
            annotation['class'] = user_input
            with open(ann_path, 'w') as f:
                json.dump(annotation, f, indent=2)
            labeled += 1

    cv2.destroyAllWindows()
    print(f'\n=== Resumen ===')
    print(f'  Etiquetados: {labeled}')
    print(f'  Saltados:    {skipped}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Etiquetado manual interactivo del dataset de señales.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Ejemplo:\n  %(prog)s ../dataset/signs'
    )
    parser.add_argument('dataset_dir', help='Directorio raíz del dataset (contiene classification/)')
    parser.add_argument('--relabel', action='store_true', help='Re-etiquetar también anotaciones que ya tienen clase asignada')
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f'Error: directorio no encontrado: {dataset_dir}')
        sys.exit(1)

    label_dataset(dataset_dir, relabel=args.relabel)


if __name__ == '__main__':
    main()