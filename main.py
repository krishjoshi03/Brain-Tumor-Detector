import streamlit as st
import requests

st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image to classify tumor type.")

uploaded_file = st.file_uploader("Choose an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True, output_format="PNG")



    if st.button("Predict"):
        # Send image to FastAPI backend
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: **{result['class']}**")
            st.info(f"Confidence: {result['confidence'] * 100:.2f}%")

        else:
            st.error("Error in prediction. Check backend logs.")
