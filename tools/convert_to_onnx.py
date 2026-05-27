import tensorflow as tf
import tf2onnx
import onnx

# Load the Keras model
model = tf.keras.models.load_model('src/navegacion_deteccion_senales/models/lane_model.keras', compile=False)

# Convert and save as ONNX with fixed batch=1 to avoid symbolic dims in CUDA arena
input_shape = model.input_shape  # e.g. (None, 224, 224, 3)
fixed_shape = (1,) + tuple(input_shape[1:])  # (1, 224, 224, 3)
input_signature = [tf.TensorSpec(shape=fixed_shape, dtype=tf.float32, name="input")]
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    output_path="src/navegacion_deteccion_senales/models/lane_model.onnx",
)

# Verificar que el modelo está bien estructurado
onnx.checker.check_model(onnx_model)

# Imprimir información legible
print("--- Inputs del Modelo ---")
for input in onnx_model.graph.input:
    print(f"Nombre: {input.name}, Tipo: {input.type.tensor_type.elem_type}, Dimensiones: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

print("\n--- Outputs del Modelo ---")
for output in onnx_model.graph.output:
    print(f"Nombre: {output.name}, Tipo: {output.type.tensor_type.elem_type}, Dimensiones: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")