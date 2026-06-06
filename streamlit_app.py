import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="E-Commerce AI Classifier",
    page_icon="🛍️",
    layout="wide"
)

# -------------------------------
# LOAD MODEL + CLASS NAMES
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/cnn_model.keras")

model = load_model()

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# -------------------------------
# IMAGE PREPROCESSING (FIX INCLUDED)
# -------------------------------
def preprocess_image(image):
    image = image.convert("RGB")   # 🔥 FIX RGBA ERROR
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------------------
# PREDICTION FUNCTION (TOP-5)
# -------------------------------
def predict(image):
    img = preprocess_image(image)
    preds = model.predict(img)

    top_indices = preds[0].argsort()[-5:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_scores = [float(preds[0][i]) for i in top_indices]

    return top_classes, top_scores

# -------------------------------
# CUSTOM CSS (CYBER UI UPGRADE)
# -------------------------------
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top, #1a0000, #000000);
    color: white;
}

/* Title */
.title {
    font-size: 52px;
    font-weight: 800;
    text-align: center;
    color: #ff1a1a;
    text-shadow: 0 0 20px #ff1a1a;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
}

/* Upload Box */
.upload-box {
    border: 2px dashed #ff1a1a;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background: rgba(255,0,0,0.05);
    transition: 0.3s;
}

.upload-box:hover {
    box-shadow: 0 0 25px #ff1a1a;
    transform: scale(1.02);
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,0,0,0.3);
    box-shadow: 0 0 20px rgba(255,0,0,0.2);
}

/* Prediction Text */
.result {
    font-size: 28px;
    font-weight: bold;
    color: #00ff99;
}

.conf {
    font-size: 18px;
    color: #ffffff;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">🛍️ E-Commerce Product Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and get instant AI-powered product prediction</div>', unsafe_allow_html=True)

st.write("")

# -------------------------------
# UPLOAD IMAGE
# -------------------------------
uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("AI is analyzing the product..."):
        top_classes, top_scores = predict(image)

    # ---------------------------
    # TOP RESULT CARD
    # ---------------------------
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="result">Top Prediction: {top_classes[0]}</div>
            <div class="conf">Confidence: {top_scores[0]*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(top_scores[0])

    # ---------------------------
    # TOP-5 BAR CHART
    # ---------------------------
    fig = go.Figure()
    fig.add_bar(
        x=top_classes,
        y=[s * 100 for s in top_scores]
    )

    fig.update_layout(
        title="Top 5 Predictions",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------
    # CONFIDENCE GAUGE
    # ---------------------------
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=top_scores[0] * 100,
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 50], 'color': "#330000"},
                {'range': [50, 80], 'color': "#ff1a1a"},
                {'range': [80, 100], 'color': "#00ff99"},
            ],
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)