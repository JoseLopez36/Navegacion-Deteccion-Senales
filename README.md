# Navegación Autónoma y Detección de Señales en AWSIM

Repositorio para el trabajo de la asignatura **Percepción en Automática y Robótica** del **MIERA** de la **Universidad de Sevilla**.

## Idea del proyecto

El objetivo es desarrollar un sistema de percepción y control básico para un vehículo autónomo simulado en **AWSIM**, conectado mediante **ROS2 Jazzy** y el ecosistema de **Autoware**.

El flujo previsto es:

1. Adquisición de imágenes desde las cámaras virtuales de AWSIM.
2. Preprocesamiento de imagen y detección de líneas de carril mediante OpenCV y transformada de Hough.
3. Estimación del error lateral del vehículo respecto al centro del carril.
4. Detección y reconocimiento de señales de tráfico mediante una CNN entrenada con TensorFlow/Keras.
5. Interpretación de señales para adaptar la actuación del vehículo, por ejemplo limitando la velocidad ante señales de velocidad máxima.
6. Publicación de comandos de control hacia Autoware/AWSIM.

Como referencias principales se usarán:

- **A. Géron**, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, O'Reilly, 3rd Edition.
- El artículo *Traffic Sign Detection and Recognition using a CNN Ensemble* como punto de partida para el módulo de detección y reconocimiento de señales.

## Equipo

- Samuel Boleslaw Locoche
- Victor Javier Granero Gil
- José Francisco López Ruiz

## Instalación

### 1. Preparar Autoware en el host

Clonar Autoware:

```bash
git clone https://github.com/autowarefoundation/autoware.git ~/autoware
cd ~/autoware
```

Preparar las herramientas de instalación del Docker de Autoware:

```bash
bash ansible/scripts/install-ansible.sh
ansible-galaxy collection install -f -r ansible-galaxy-requirements.yaml
ansible-playbook autoware.dev_env.install_docker -K
```

Más info: [Documentación de instalación de Autoware con Docker](https://autowarefoundation.github.io/autoware-documentation/main/installation/autoware/docker-installation/)

### 2. Descargar los datos de planificación

Descargar el mapa `sample-map-planning` y los modelos ML en `~/autoware_data`:

```bash
mkdir ~/autoware_data
cd ~/autoware_data
mkdir maps ml_models
ansible-playbook autoware.dev_env.install_dev_env --tags demo_artifacts --ask-become-pass
ansible-playbook autoware.dev_env.install_dev_env --tags ml_models --ask-become-pass
```

El script deja los datos en:

```text
~/autoware_data/maps/sample-map-planning/
~/autoware_data/ml_models/
```

Más info: [Documentación de planning simulation](https://autowarefoundation.github.io/autoware-documentation/main/demos/planning-sim/)

### 3. Descargar y ejecutar el contenedor

Descargar la imagen de Autoware para ROS2 Jazzy:

```bash
docker pull ghcr.io/autowarefoundation/autoware:universe-cuda-jazzy
```

Ejecutar el contenedor:

```bash
./tools/run_docker.sh
```

Probar el simulador de planificación de Autoware:

```bash
source ~/autoware/install/setup.bash
ros2 launch autoware_launch planning_simulator.launch.xml map_path:=$HOME/autoware_data/maps/sample-map-planning vehicle_model:=sample_vehicle sensor_model:=sample_sensor_kit
```

### 4. Instalar y ejecutar AWSIM

AWSIM permite ejecutar una simulación fotorrealista conectada con Autoware. Requiere una GPU NVIDIA RTX y drivers NVIDIA compatibles. Descargar AWSIM Demo y el mapa de Shinjuku en [AWSIM Quick Start Demo](https://tier4.github.io/AWSIM/GettingStarted/QuickStartDemo/).

Ejecutar AWSIM desde el host:

```bash
./AWSIM-demo.x86_64 --json_path AWSIM-config.json
```

Lanzar Autoware conectado a AWSIM:

```bash
xhost +local:docker

cd tools/
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm awsim
```

Más info: [AWSIM Quick Start Demo](https://tier4.github.io/AWSIM/GettingStarted/QuickStartDemo/)

## Puesta en marcha

Compilar el paquete ROS2 dentro del contenedor de desarrollo:

```bash
./tools/run_docker.sh
```

```bash
cd /home/aw/workspace
rm -rf build/navegacion_deteccion_senales install/navegacion_deteccion_senales log
colcon build --symlink-install --packages-select navegacion_deteccion_senales
source install/setup.bash
```

Con AWSIM ya ejecutándose, lanzar el sistema de navegación y percepción usando Docker Compose:

```bash
cd tools/
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose run --rm navegacion_deteccion_senales
```

El servicio `navegacion_deteccion_senales` de `tools/docker-compose.yaml` monta el workspace, activa `/opt/autoware/setup.bash`, activa `install/setup.bash` y ejecuta `ros2 launch navegacion_deteccion_senales run.launch.py`.

El nodo se suscribe a la cámara de AWSIM en `/sensing/camera/traffic_light/image_raw`, publica comandos de control en `/control/command/control_cmd` y genera una imagen de depuración en `~/debug_image`.