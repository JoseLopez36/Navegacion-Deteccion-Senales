#!/usr/bin/env python3
"""
Script para anotar manualmente bounding boxes de señales de tráfico.

Uso:
  python3 annotate_signs_dataset.py <dataset_dir> [options]

Ejemplos:
  # Anotar imágenes nuevas
  python3 annotate_signs_dataset.py dataset/signs

  # Revisar y corregir anotaciones existentes
  python3 annotate_signs_dataset.py dataset/signs --review

  # Empezar desde una imagen específica
  python3 annotate_signs_dataset.py dataset/signs --start 15
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

# Clases disponibles para anotación
CLASSES = [
    "stop",           # 0 - Señal de stop
    "give_way",       # 1 - Ceda el paso
    "no_entry",       # 2 - Prohibido el paso
    "one_way",        # 3 - Sentido único
    "turn_left",      # 4 - Giro izquierda
    "turn_right",     # 5 - Giro derecha
    "straight",       # 6 - Sigue recto
    "no_turn_left",   # 7 - Prohibido girar izquierda
    "no_turn_right",  # 8 - Prohibido girar derecha
    "speed_limit",    # 9 - Límite de velocidad
    "parking",        # 10 - Parking
    "crosswalk",      # 11 - Paso de peatones
]

# Colores para cada clase (en BGR)
CLASS_COLORS = [
    (0, 0, 255),      # stop - rojo
    (0, 165, 255),    # give_way - naranja
    (0, 0, 128),      # no_entry - rojo oscuro
    (255, 0, 0),      # one_way - azul
    (0, 255, 0),      # turn_left - verde
    (255, 255, 0),    # turn_right - cyan
    (255, 0, 255),    # straight - magenta
    (0, 128, 128),    # no_turn_left - marrón
    (128, 128, 0),    # no_turn_right - azul grisáceo
    (128, 0, 128),    # speed_limit - púrpura
    (255, 255, 255),  # parking - blanco
    (128, 128, 128),  # crosswalk - gris
]


class AnnotationTool:
    def __init__(self, dataset_dir, review_mode=False, start_idx=0):
        self.dataset_path = Path(dataset_dir)
        self.images_dir = self.dataset_path / 'images'
        self.annotations_dir = self.dataset_path / 'annotations'
        self.review_mode = review_mode
        self.current_idx = start_idx
        
        # Crear directorio de anotaciones si no existe
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado del dibujo
        self.drawing = False
        self.start_point = None
        self.current_bbox = None
        self.detections = []
        self.selected_class_idx = 0
        
        # Cargar imágenes
        self.image_files = self._load_image_list()
        if not self.image_files:
            print("Error: No se encontraron imágenes")
            sys.exit(1)
            
        if self.current_idx >= len(self.image_files):
            self.current_idx = 0
            
        # Configurar ventana
        self.window_name = "Annotation Tool - Presiona 'h' para ayuda"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
    def _load_image_list(self):
        """Carga la lista de imágenes del directorio."""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(self.images_dir.glob(ext))
        return sorted(image_files)
    
    def _load_annotations(self, image_path):
        """Carga anotaciones existentes si las hay."""
        annotation_path = self.annotations_dir / (image_path.stem + '.json')
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                return data.get('detections', [])
        return []
    
    def _save_annotations(self, image_path):
        """Guarda las anotaciones en formato JSON."""
        annotation_path = self.annotations_dir / (image_path.stem + '.json')
        
        annotation_data = {
            'image': image_path.name,
            'image_width': self.image.shape[1],
            'image_height': self.image.shape[0],
            'num_detections': len(self.detections),
            'detections': self.detections
        }
        
        with open(annotation_path, 'w') as f:
            json.dump(annotation_data, f, indent=2)
        print(f"Guardado: {annotation_path}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del ratón."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_bbox = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                x1, y1 = self.start_point
                self.current_bbox = (min(x1, x), min(y1, y), abs(x - x1), abs(y - y1))
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_bbox:
                x, y, w, h = self.current_bbox
                if w > 10 and h > 10:  # Mínimo tamaño
                    class_name = CLASSES[self.selected_class_idx]
                    color = CLASS_COLORS[self.selected_class_idx]
                    self.detections.append({
                        'class': class_name,
                        'confidence': 1.0,
                        'bbox': [int(x), int(y), int(w), int(h)]
                    })
                    print(f"Añadida: {class_name} en ({x}, {y}, {w}, {h})")
            self.drawing = False
            self.current_bbox = None
    
    def _draw_overlay(self):
        """Dibuja la imagen con todas las anotaciones."""
        overlay = self.image.copy()
        
        # Dibujar detecciones guardadas
        for det in self.detections:
            x, y, w, h = det['bbox']
            class_name = det['class']
            class_idx = CLASSES.index(class_name) if class_name in CLASSES else 0
            color = CLASS_COLORS[class_idx]
            
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            label = f"{class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_y = y - 10 if y - 10 > 10 else y + 20
            
            cv2.rectangle(
                overlay,
                (x, label_y - label_size[1] - 4),
                (x + label_size[0], label_y + 4),
                color, -1
            )
            cv2.putText(
                overlay, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Dibujar bbox en progreso
        if self.current_bbox:
            x, y, w, h = self.current_bbox
            color = CLASS_COLORS[self.selected_class_idx]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.line(overlay, (x, y), (x + w, y + h), color, 1)
            cv2.line(overlay, (x + w, y), (x, y + h), color, 1)
        
        # Panel de información
        info_y = 30
        cv2.putText(
            overlay, f"Imagen: {self.current_idx + 1}/{len(self.image_files)}",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        info_y += 25
        cv2.putText(
            overlay, f"Detecciones: {len(self.detections)}",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        info_y += 25
        cv2.putText(
            overlay, f"Clase: [{self.selected_class_idx}] {CLASSES[self.selected_class_idx]}",
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLASS_COLORS[self.selected_class_idx], 2
        )
        
        return overlay
    
    def _show_help(self):
        """Muestra ventana de ayuda."""
        help_text = [
            "CONTROLES:",
            "",
            "Ratón:",
            "  Click + arrastrar - Dibujar bounding box",
            "",
            "Teclas:",
            "  0-9, q,w,e,r... - Cambiar clase",
            "  u / Backspace   - Eliminar última anotación",
            "  c               - Limpiar todas las anotaciones",
            "  s               - Guardar anotaciones",
            "  n / →           - Siguiente imagen",
            "  p / ←           - Imagen anterior",
            "  h               - Mostrar esta ayuda",
            "  q / ESC         - Salir",
            "",
            "CLASES:",
        ]
        for i, cls in enumerate(CLASSES):
            help_text.append(f"  {i}: {cls}")
        
        # Crear imagen de ayuda
        line_height = 20
        width = 500
        height = len(help_text) * line_height + 40
        help_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        y = 30
        for line in help_text:
            if line.startswith("  ") and ": " in line and line[2:3].isdigit():
                idx = int(line[2:line.index(":")])
                color = CLASS_COLORS[idx] if idx < len(CLASS_COLORS) else (255, 255, 255)
            elif line in ["CONTROLES:", "Teclas:", "Ratón:", "CLASES:"]:
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)
            cv2.putText(help_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line_height
            
        cv2.imshow("Ayuda", help_img)
    
    def _get_class_from_key(self, key):
        """Obtiene el índice de clase desde una tecla."""
        key_map = {
            ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3, ord('4'): 4,
            ord('5'): 5, ord('6'): 6, ord('7'): 7, ord('8'): 8, ord('9'): 9,
            ord('q'): 10, ord('w'): 11,
        }
        return key_map.get(key)
    
    def run(self):
        """Bucle principal de la aplicación."""
        print(f"Total de imágenes: {len(self.image_files)}")
        print("Presiona 'h' para ver la ayuda")
        
        while self.current_idx < len(self.image_files):
            image_path = self.image_files[self.current_idx]
            print(f"\n[{self.current_idx + 1}/{len(self.image_files)}] {image_path.name}")
            
            # Cargar imagen
            self.image = cv2.imread(str(image_path))
            if self.image is None:
                print(f"Error: No se pudo cargar {image_path}")
                self.current_idx += 1
                continue
            
            # Cargar anotaciones existentes
            self.detections = self._load_annotations(image_path)
            if self.detections:
                print(f"Cargadas {len(self.detections)} anotaciones existentes")
            
            # Redimensionar ventana si es necesario
            max_height = 800
            if self.image.shape[0] > max_height:
                scale = max_height / self.image.shape[0]
                new_width = int(self.image.shape[1] * scale)
                cv2.resizeWindow(self.window_name, new_width, max_height)
            else:
                cv2.resizeWindow(self.window_name, self.image.shape[1], self.image.shape[0])
            
            while True:
                # Dibujar y mostrar
                overlay = self._draw_overlay()
                cv2.imshow(self.window_name, overlay)
                
                # Esperar tecla
                key = cv2.waitKey(1) & 0xFF
                
                # Cambiar clase
                class_idx = self._get_class_from_key(key)
                if class_idx is not None and class_idx < len(CLASSES):
                    self.selected_class_idx = class_idx
                    print(f"Clase seleccionada: {CLASSES[class_idx]}")
                    
                # Eliminar última anotación
                elif key == ord('u') or key == 127:  # u o Backspace
                    if self.detections:
                        removed = self.detections.pop()
                        print(f"Eliminada: {removed['class']}")
                        
                # Limpiar todas
                elif key == ord('c'):
                    self.detections = []
                    print("Anotaciones limpiadas")
                    
                # Guardar
                elif key == ord('s'):
                    self._save_annotations(image_path)
                    
                # Siguiente imagen
                elif key == ord('n') or key == 83:  # n o flecha derecha
                    self._save_annotations(image_path)
                    self.current_idx += 1
                    break
                    
                # Imagen anterior
                elif key == ord('p') or key == 81:  # p o flecha izquierda
                    if self.current_idx > 0:
                        self._save_annotations(image_path)
                        self.current_idx -= 1
                        break
                        
                # Ayuda
                elif key == ord('h'):
                    self._show_help()
                    
                # Cerrar ayuda
                elif key == ord('H') or key == 27:  # ESC
                    cv2.destroyWindow("Ayuda")
                    
                # Salir
                elif key == ord('q') or key == 27:
                    self._save_annotations(image_path)
                    print("\nSaliendo...")
                    cv2.destroyAllWindows()
                    return
        
        print("\nTodas las imágenes procesadas")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Herramienta de anotación manual de señales de tráfico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controles básicos:
  Click + arrastrar   - Dibujar bounding box
  0-9, q, w           - Seleccionar clase
  u / Backspace       - Eliminar última anotación
  c                   - Limpiar todas
  s                   - Guardar
  n / →               - Siguiente imagen
  p / ←               - Imagen anterior
  h                   - Ayuda
  q / ESC             - Salir
"""
    )
    parser.add_argument('dataset_dir', help='Directorio del dataset (debe tener subcarpeta images/)')
    parser.add_argument('--review', action='store_true', help='Modo revisión: cargar anotaciones existentes')
    parser.add_argument('--start', type=int, default=0, help='Índice de imagen inicial (0-based)')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    images_dir = dataset_path / 'images'
    
    if not images_dir.exists():
        print(f"Error: No se encontró el directorio de imágenes: {images_dir}")
        print("El dataset debe tener la estructura:")
        print("  dataset/")
        print("    images/")
        print("    annotations/")
        sys.exit(1)
    
    tool = AnnotationTool(
        args.dataset_dir,
        review_mode=args.review,
        start_idx=args.start
    )
    tool.run()


if __name__ == '__main__':
    main()