import tensorflow as tf
import tf_keras as keras

# Učitaj koristeći tf_keras
model = keras.models.load_model("./outputs/best_person_detection.keras")

# Konvertiraj u TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("person_detection.tflite", "wb") as f:
    f.write(tflite_model)
print("🎯 TFLite model spremljen!")