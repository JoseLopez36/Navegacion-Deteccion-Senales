import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    mu = y_pred[:, :, :, 0]
    y_pred_f = K.flatten(mu)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def recall_smooth(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection / (K.sum(y_true_f) + K.epsilon()))


def precision_smooth(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection / (K.sum(y_pred_f) + K.epsilon())


def accuracy(y_true, y_pred):
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    true_positives = K.sum(K.round(K.clip(y_true_f * y_pred_f, 0, 1)))
    true_negatives = K.sum(K.round(K.clip((1 - y_true_f) * (1 - y_pred_f), 0, 1)))
    total_pixels = K.cast(tf.size(y_true_f), K.floatx())
    accuracy_value = (true_positives + true_negatives) / total_pixels
    return accuracy_value


def load_model(model_path="model_VGG.keras"):
    """Carga el modelo Keras con los custom objects necesarios."""
    try:
        return tf.keras.models.load_model(
            model_path,
            custom_objects={
                "dice_loss": dice_loss,
                "dice_coefficent": dice_coefficient,
                "dice_coefficient": dice_coefficient,
                "precision_smooth": precision_smooth,
                "recall_smooth": recall_smooth,
                "accuracy": accuracy
            }
        )
    except (TypeError, ModuleNotFoundError, ValueError) as e:
        raise RuntimeError(
            f"No se pudo cargar el modelo desde {model_path}. "
            f"Asegurate de que la version de TensorFlow/Keras coincide "
            f"con la usada para guardar el modelo .keras. Error: {e}"
        ) from e

def _sharpen_image(image):
    """Aplica filtro de sharpening a la imagen."""
    kernel = tf.constant([[0., -1., 0.],
                          [-1., 5., -1.],
                          [0., -1., 0.]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    channels = tf.split(image, num_or_size_splits=3, axis=-1)
    sharpened_channels = []
    for c in channels:
        c_sharp = tf.nn.conv2d(tf.expand_dims(c, axis=0), kernel, strides=1, padding="SAME")
        sharpened_channels.append(tf.squeeze(c_sharp, axis=0))
    image = tf.concat(sharpened_channels, axis=-1)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_frame(frame, size=(224, 224)):
    """
    Preprocesa un frame de OpenCV (numpy array) para el modelo.
    """
    # Convertir BGR a RGB y a tensor
    image = tf.convert_to_tensor(frame, dtype=tf.float32)
    image = tf.image.resize(image, size)
    image = image / 255.0
    image = _sharpen_image(image)
    return image


def mask_to_array(predicted_mask):
    """Convierte máscara de predicción a array numpy 2D float32."""
    mask = np.array(predicted_mask)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.float32)


def predict_lane(image, model):
    """Predice la máscara de carril para una imagen preprocesada."""
    batch = tf.expand_dims(image, axis=0)
    pred_mask = model(batch, training=False)
    return tf.math.round(pred_mask[0])