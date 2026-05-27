import tensorflow as tf
import tf2onnx

# Load the Keras model
model = tf.keras.models.load_model('src/navegacion_deteccion_senales/models/lane_model.keras', compile=False)

# Convert and save as ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, output_path="src/navegacion_deteccion_senales/models/lane_model.onnx")