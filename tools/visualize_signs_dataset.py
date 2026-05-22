#!/usr/bin/env python3
"""
Script para visualizar el dataset de señales con bounding boxes.

Uso:
  python3 visualize_signs_dataset.py <dataset_dir> [options]

Ejemplos:
  # Mostrar todas las imágenes
  python3 visualize_signs_dataset.py dataset/signs

  # Mostrar solo imágenes con múltiples señales
  python3 visualize_signs_dataset.py dataset/signs --min-signs 2

  # Guardar imágenes anotadas
  python3 visualize_signs_dataset.py dataset/signs --output dataset/signs/visualized

  # Modo slideshow con delay de 2 segundos
  python3 visualize_signs_dataset.py dataset/signs --slideshow --delay 2000
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np


def draw_detections(image, detections, color=(0, 255, 0), thickness=2):
    """Dibuja bounding boxes en la imagen."""
    img_copy = image.copy()
    for det in detections:
        x, y, w, h = det['bbox']
        class_name = det.get('class', 'unknown')
        confidence = det.get('confidence', 1.0)
        
        # Dibujar bounding box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, thickness)
        
        # Dibujar etiqueta
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_y = y - 10 if y - 10 > 10 else y + 20
        
        cv2.rectangle(
            img_copy, 
            (x, label_y - label_size[1] - 4), 
            (x + label_size[0], label_y + 4), 
            color, 
            -1
        )
        cv2.putText(
            img_copy, label, (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return img_copy


def process_image(image_path, annotation_path, args):
    """Procesa una imagen con sus anotaciones."""
    # Cargar imagen
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: No se pudo cargar {image_path}")
        return False
    
    # Cargar anotaciones
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    
    detections = annotation.get('detections', [])
    num_signs = len(detections)
    
    # Filtrar por número mínimo de señales
    if args.min_signs and num_signs < args.min_signs:
        return False
    
    # Dibujar detecciones
    vis_image = draw_detections(image, detections)
    
    # Añadir info de contador
    info_text = f"Senales: {num_signs}"
    cv2.putText(
        vis_image, info_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    
    # Guardar o mostrar
    if args.output:
        output_path = Path(args.output) / image_path.name
        cv2.imwrite(str(output_path), vis_image)
        print(f"Guardado: {output_path}")
    else:
        # Mostrar en ventana
        window_name = "Dataset Visualization"
        cv2.imshow(window_name, vis_image)
        
        if args.slideshow:
            key = cv2.waitKey(args.delay)
        else:
            key = cv2.waitKey(0)
        
        if key == ord('q') or key == 27:  # 'q' o ESC
            return True  # Señal de salir
    
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Visualiza el dataset de señales de tráfico con bounding boxes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Teclas:
  q / ESC  - Salir
  ESPACIO  - Siguiente imagen (modo manual)

Ejemplos:
  %(prog)s ../dataset
  %(prog)s ../dataset --min-signs 2
  %(prog)s ../dataset --output ../dataset/visualized
  %(prog)s ../dataset --slideshow --delay 1000
"""
    )
    parser.add_argument('dataset_dir', help='Directorio del dataset')
    parser.add_argument('--min-signs', type=int, help='Mostrar solo imágenes con al menos N señales')
    parser.add_argument('--output', help='Directorio para guardar imágenes anotadas (en lugar de mostrar)')
    parser.add_argument('--slideshow', action='store_true', help='Modo slideshow automático')
    parser.add_argument('--delay', type=int, default=1000, help='Delay en ms entre imágenes (modo slideshow)')
    parser.add_argument('--limit', type=int, help='Limitar número de imágenes a procesar')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    images_dir = dataset_path / 'images'
    annotations_dir = dataset_path / 'annotations'
    
    if not images_dir.exists():
        print(f"Error: No se encontró el directorio de imágenes: {images_dir}")
        sys.exit(1)
    
    if not annotations_dir.exists():
        print(f"Error: No se encontró el directorio de anotaciones: {annotations_dir}")
        sys.exit(1)
    
    # Crear directorio de salida si es necesario
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Obtener lista de imágenes
    image_files = sorted(images_dir.glob('*.jpg'))
    if not image_files:
        image_files = sorted(images_dir.glob('*.png'))
    
    print(f"Encontradas {len(image_files)} imágenes")
    
    processed = 0
    for i, image_path in enumerate(image_files):
        if args.limit and i >= args.limit:
            break
        
        # Buscar archivo de anotación correspondiente
        annotation_path = annotations_dir / (image_path.stem + '.json')
        
        if not annotation_path.exists():
            print(f"Advertencia: No se encontró anotación para {image_path.name}")
            continue
        
        should_quit = process_image(image_path, annotation_path, args)
        processed += 1
        
        if should_quit:
            break
    
    cv2.destroyAllWindows()
    print(f"\nProcesadas {processed} imágenes")


if __name__ == '__main__':
    main()