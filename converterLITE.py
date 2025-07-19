import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("data/cat_dog_classifier.keras")

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("cat_dog_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Converted and saved as cat_dog_classifier.tflite")
