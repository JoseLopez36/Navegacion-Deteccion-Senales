# Navegación Autónoma y Detección de Señales en CARLA

Repositorio para el trabajo de la asignatura **Percepción en Automática y Robótica** del **MIERA** de la **Universidad de Sevilla**.

## Idea del proyecto

El objetivo es desarrollar un sistema de percepción y control básico para un vehículo autónomo simulado en **CARLA Simulator**, conectado mediante **ROS2 Humble**.

El flujo previsto es:

1. Adquisición de imágenes desde las cámaras virtuales de CARLA.
2. Preprocesamiento de imagen y detección de líneas de carril mediante un modelo de segmentación semántica (U-Net con backbone VGG) entrenado con TensorFlow/Keras.
3. Estimación del error lateral del vehículo respecto al centro del carril a partir de la máscara predicha.
4. Detección y reconocimiento de señales de tráfico mediante una CNN.
5. Interpretación de señales para adaptar la actuación del vehículo, por ejemplo limitando la velocidad ante señales de velocidad máxima.
6. Publicación de comandos de control hacia CARLA.

Como referencias principales se usarán:

- **A. Géron**, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, O'Reilly, 3rd Edition.
- El artículo *Traffic Sign Detection and Recognition using a CNN Ensemble* como punto de partida para el módulo de detección y reconocimiento de señales.

## Equipo

- Samuel Boleslaw Locoche.
- Victor Javier Granero Gil.
- José Francisco López Ruiz.

## Instalación

### Instalar CARLA Simulator con Docker

Instalar NVIDIA Container Toolkit:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Descargar y ejecutar CARLA Simulator:

```bash
# Descargar la imagen de CARLA 0.9.15
docker pull carlasim/carla:0.9.15

# Ejecutar CARLA
xhost +local:docker
docker run --rm --privileged --gpus all --net=host \
  -e DISPLAY=$DISPLAY \
  -e XDG_RUNTIME_DIR=/tmp/runtime-carla \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --user $(id -u):$(id -g) \
  --workdir /home/carla \
  -it carlasim/carla:0.9.15 \
  ./CarlaUE4.sh -windowed -carla-rpc-port=2001 -nosound

# Ejecutar CARLA en modo headless
docker run --rm --privileged --gpus all --net=host \
  --user $(id -u):$(id -g) \
  --workdir /home/carla \
  -it carlasim/carla:0.9.15 \
  ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2001 -nosound
```

Más info: [Guía de instalación de CARLA con Docker](https://medium.com/aimonks/downloading-carla-simulator-with-docker-on-ubuntu-22-04-in-2025-220904de2942)

## Puesta en marcha

### Construir la imagen del contenedor ROS2

Desde la raíz del repositorio:

```bash
docker build -t navegacion_deteccion_senales docker/
```

### Lanzar con Docker Compose

Con CARLA ya corriendo, levantar el bridge y el sistema de navegación en un solo comando desde la raíz del repositorio:

```bash
xhost +local:docker
docker compose -f tools/docker-compose.yaml up
```

Esto arranca tres contenedores en paralelo:

| Servicio | Contenedor | Descripción |
|---|---|---|
| `ros_bridge` | `carla_ros_bridge` | Bridge CARLA → ROS2 + spawn del ego-vehicle |
| `navegacion` | `navegacion_deteccion_senales` | Compila y lanza el paquete del proyecto |
| `foxglove` | `foxglove_bridge` | WebSocket bridge en `ws://localhost:8765` para Foxglove Studio |

Para lanzarlos en terminales separadas y ver los logs independientemente:

```bash
docker compose -f tools/docker-compose.yaml up ros_bridge
docker compose -f tools/docker-compose.yaml up navegacion
docker compose -f tools/docker-compose.yaml up foxglove
```

### Ejecutar comandos en los contenedores

Abrir una shell interactiva en cualquiera de los tres contenedores en ejecución:

```bash
docker exec -it carla_ros_bridge bash
docker exec -it navegacion_deteccion_senales bash
docker exec -it foxglove_bridge bash
```

### Visualización con Foxglove Studio

Foxglove Studio es una alternativa a rviz2 que corre en el **host** (o en el navegador) y se conecta al contenedor sin necesidad de reenvío X11.

#### Conexión

1. Abre [Foxglove Studio](https://studio.foxglove.dev) en el navegador o la app de escritorio.
2. Selecciona **Open connection → Foxglove WebSocket**.
3. Introduce la URL:
   ```
   ws://172.17.0.1:8765
   ```
   > `172.17.0.1` es la IP del host desde dentro de la red Docker bridge. El puerto `8765` es el que expone el servicio `foxglove_bridge` del compose.

#### Importar layout

Para cargar el layout predefinido con las dos cámaras (`rgb_view` y `rgb_front`):

1. En Foxglove Studio, ve a **View → Import layout from file** (o el icono de layout en la barra lateral).
2. Selecciona el fichero `tools/foxglove.json` de este repositorio.

El layout carga dos paneles de imagen:
- **Izquierda**: `/carla/ego_vehicle/rgb_view/image` — vista exterior del vehículo
- **Derecha**: `/carla/ego_vehicle/rgb_front/image` — cámara frontal (detección de señales)

## Generación de Dataset

El paquete incluye dos nodos de recolección que funcionan en modo conducción manual (sin `vehicle_control_node`). Controla el vehículo con `W/A/S/D` desde CARLA (`B` para activar el modo manual).

Ambos usan la **segmentación semántica de CARLA** como ground truth perfecto.

### Dataset de señales

```bash
xhost +local:docker
docker compose -f tools/docker-compose-sign-dataset.yaml up
```

- **Nodo**: `sign_dataset_node` — detecta señales (color amarillo en semántica), extrae bounding boxes y guarda imagen RGB + JSON de anotaciones.
- **Salida**: `dataset/signs/images/` y `dataset/signs/annotations/`

### Dataset de carriles

```bash
xhost +local:docker
docker compose -f tools/docker-compose-lane-dataset.yaml up
```

- **Nodo**: `lane_dataset_node` — detecta marcas viales en la máscara semántica, la remap al espacio RGB y guarda imagen RGB + máscara binaria de ground truth.
- **Salida**: `dataset/lanes/images/` y `dataset/lanes/masks/`

### Visualización

```bash
# Señales (bounding boxes sobre RGB)
python3 tools/visualize_signs_dataset.py dataset/signs
python3 tools/visualize_signs_dataset.py dataset/signs --min-signs 2 --slideshow --delay 1000
python3 tools/visualize_signs_dataset.py dataset/signs --output dataset/signs/visualized

# Carriles (máscara superpuesta sobre RGB)
python3 tools/visualize_lanes_dataset.py dataset/lanes
python3 tools/visualize_lanes_dataset.py dataset/lanes --slideshow --delay 1000
python3 tools/visualize_lanes_dataset.py dataset/lanes --output dataset/lanes/visualized
```

**Ejecutar en Docker:**

```bash
xhost +local:docker

# Señales
docker run --rm -it --network host -e DISPLAY=${DISPLAY} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/dataset/signs:/dataset:ro \
  -v $(pwd)/tools:/tools:ro \
  navegacion_deteccion_senales \
  python3 /tools/visualize_signs_dataset.py /dataset

# Carriles
docker run --rm -it --network host -e DISPLAY=${DISPLAY} \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $(pwd)/dataset/lanes:/dataset:ro \
  -v $(pwd)/tools:/tools:ro \
  navegacion_deteccion_senales \
  python3 /tools/visualize_lanes_dataset.py /dataset
```