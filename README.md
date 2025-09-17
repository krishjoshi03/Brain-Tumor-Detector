# 🧠 Brain Tumor Classification (FastAPI + Streamlit)

This project is a **Brain Tumor Classification System** built using **PyTorch, FastAPI, and Streamlit**.  
It allows users to upload an MRI scan image and get a prediction of whether a brain tumor is present or not.  
The backend is powered by **FastAPI** for model inference, and the frontend is created using **Streamlit** for an easy-to-use UI.  

---

##  Features
- Upload an MRI scan image.
- Get predictions with confidence scores.
- Interactive web interface.
- Modular code structure (FastAPI for backend, Streamlit for frontend).
- Model trained with **CNN / EfficientNet b0 / ResNet 50**.

---

![UI Preview](Demo.png)


##  Project Structure
- Tumor Recognization
- model
- main.py
- prediction_helper
- requirements.txt 
- README.md 

### Clone Repository
```bash
    git clone https://github.com/your-username/brain-tumor-classification.git
    cd brain-tumor-classification
```

### Install Dependencies
```bash
    pip install -r requirements.txt
```

### Running The Project (Fastapi Backend)
```bash
    uvicorn app:app --reload
```

### Running Frontend (Streamlit app)
```bash
    streamlit run main.py
```

## Usage
 - Open the Streamlit app.
 - Upload an MRI scan image.
 - Click on Predict.
 - The model will display:
 - Predicted Class (e.g., Tumor / No Tumor)
 - Confidence Score (%)


