import tensorflow as tf

keras_model = tf.keras.models.load_model("model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

# Make sure to keep float32 don't use int8
converter.optimizations = []  # No optimizing
converter.target_spec.supported_types = [tf.float32]

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
