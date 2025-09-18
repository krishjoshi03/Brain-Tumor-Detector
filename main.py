import streamlit as st
from PIL import Image
import prediction_helper  # import our helper file

st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image to classify tumor type.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Predict"):
        pred_class, confidence = prediction_helper.predict_image(image)
        st.success(f"Prediction: **{pred_class}**")
        st.info(f"Confidence: {confidence * 100:.2f}%")
