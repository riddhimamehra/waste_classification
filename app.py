import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image


model = tf.keras.models.load_model("waste_classifier.keras")

with open("class_names.json", "r") as f:
    class_names = json.load(f)


st.title("Waste Classification")
st.write("Upload an image and the model will classify it into one of 7 waste categories-Glass, Metal, Paper, Plastic, Cardboard, Battery, Clothes.")


uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    if st.button("🔍 Predict"):

        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index] * 100

        st.subheader("Prediction")
        st.success(f"♻️ {predicted_class}")
        st.write(f"Confidence: **{confidence:.2f}%**")