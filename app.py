import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EcoSort — Waste Classifier",
    page_icon="♻️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #a7f3d0;
        text-align: center;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #065f46;
    }
    .result-confidence {
        font-size: 1rem;
        color: #047857;
        margin-top: 0.2rem;
    }
    .category-pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 3px;
        border-radius: 999px;
        background: #f3f4f6;
        font-size: 0.85rem;
        color: #374151;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model + assets
# ---------------------------------------------------------------------------
CATEGORY_ICONS = {
    "Battery": "🔋",
    "Cardboard": "📦",
    "Clothes": "👕",
    "Glass": "🍾",
    "Metal": "🥫",
    "Paper": "📄",
    "Plastic": "🧴",
}


@st.cache_resource(show_spinner=False)
def load_assets():
    model = tf.keras.models.load_model("waste_classifier.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_assets()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About EcoSort")
    st.write(
        "EcoSort classifies waste images into 7 categories using a "
        "fine-tuned EfficientNetB0 model, achieving 90% validation accuracy."
    )
    st.markdown("**Categories:**")
    st.markdown(
        " ".join(
            f'<span class="category-pill">{CATEGORY_ICONS.get(c, "")} {c}</span>'
            for c in class_names
        ),
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Built with TensorFlow, EfficientNetB0, and Streamlit.")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">EcoSort</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a photo of a waste item and let EcoSort sort it for you</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        predict_clicked = st.button("Predict Waste Type", use_container_width=True)

    if predict_clicked:
        with st.spinner("Analyzing image..."):
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized)
            image_array = np.expand_dims(image_array, axis=0)

            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100
            icon = CATEGORY_ICONS.get(predicted_class)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">{icon} {predicted_class}</div>
                <div class="result-confidence">Confidence: {confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Confidence across all categories")
        chart_df = pd.DataFrame({
            "Category": [f"{CATEGORY_ICONS.get(c, '')} {c}" for c in class_names],
            "Confidence (%)": [float(p) * 100 for p in predictions],
        }).set_index("Category")
        st.bar_chart(chart_df, height=280)

else:
    st.info("Upload an image to get started — supports JPG, JPEG, and PNG.")
