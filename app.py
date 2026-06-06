import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ----------------------------
# LOAD MODEL
# ----------------------------
model = tf.keras.models.load_model("models/cnn_model.keras")

# ----------------------------
# LOAD CLASS NAMES
# ----------------------------
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------------------
# PREPROCESS IMAGE
# ----------------------------
def preprocess(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict(image):
    img = preprocess(image)
    preds = model.predict(img)[0]

    top_index = np.argmax(preds)
    label = class_names[top_index]
    confidence = float(preds[top_index]) * 100

    return {
        "Prediction": label,
        "Confidence (%)": round(confidence, 2)
    }

# ----------------------------
# GRADIO UI
# ----------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.JSON(),
    title="🛍️ E-Commerce Product Classifier",
    description="Upload an image and the CNN model will predict the product category."
)

demo.launch()