# Navegación Autónoma y Detección de Señales en CARLA

Repositorio para el trabajo de la asignatura **Percepción en Automática y Robótica** del **MIERA** de la **Universidad de Sevilla**.

## Idea del proyecto

El objetivo es desarrollar un sistema de percepción y control básico para un vehículo autónomo simulado en **CARLA Simulator**, conectado mediante **ROS2 Humble**.

El flujo previsto es:

1. Adquisición de imágenes desde las cámaras virtuales de CARLA.
2. Preprocesamiento de imagen y detección de líneas de carril mediante OpenCV y transformada de Hough.
3. Estimación del error lateral del vehículo respecto al centro del carril.
4. Detección y reconocimiento de señales de tráfico mediante una CNN entrenada con PyTorch.
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
docker build -t ros2-carla:humble docker/
```

### Lanzar con Docker Compose

Con CARLA ya corriendo, levantar el bridge y el sistema de navegación en un solo comando desde la raíz del repositorio:

```bash
docker compose -f tools/docker-compose-carla.yaml up
```

Esto arranca tres contenedores en paralelo:

| Servicio | Contenedor | Descripción |
|---|---|---|
| `ros_bridge` | `carla_ros_bridge` | Bridge CARLA → ROS2 + spawn del ego-vehicle |
| `navegacion` | `navegacion_deteccion_senales` | Compila y lanza el paquete del proyecto |
| `foxglove` | `foxglove_bridge` | WebSocket bridge en `ws://localhost:8765` para Foxglove Studio |

Para lanzarlos en terminales separadas y ver los logs independientemente:

```bash
docker compose -f tools/docker-compose-carla.yaml up ros_bridge
docker compose -f tools/docker-compose-carla.yaml up navegacion
docker compose -f tools/docker-compose-carla.yaml up foxglove
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