import tensorflow as tf

root_dir = "/Data/Python/AI/level_detection"
data_root = f"{root_dir}/dataset"
model_path = "model.keras"
model_path_2 = "model.tflite"

model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Make sure to keep float32 don't use int8
converter.optimizations = []  # No optimizing
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

with open(model_path_2, "wb") as f:
    f.write(tflite_model)
