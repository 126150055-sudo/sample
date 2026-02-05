import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (160, 160)
MODEL_PATH = "dataset/doctor_lung_disease_model.h5"

st.set_page_config(
    page_title="AI Lung Disease Detection",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# DOCTOR DECISION SYSTEM
# -----------------------------
def doctor_final_report(prob):
    if prob >= 0.90:
        return "Severe Pneumonia", "High", \
               "Immediate hospitalization and oxygen therapy required."
    elif prob >= 0.70:
        return "Moderate Pneumonia", "Medium", \
               "Antibiotic treatment and close monitoring advised."
    elif prob >= 0.50:
        return "Mild Lung Infection", "Low", \
               "Home care with follow-up X-ray suggested."
    else:
        return "Normal Lung", "Normal", \
               "No abnormal lung disease detected."

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🫁 AI-Based Lung Disease Detection System")
st.subheader("PG Major Project – Doctor Decision Support")

st.markdown("""
Upload a **Chest X-ray image** to detect pneumonia and receive a  
**doctor-style clinical report**.
""")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Analyzing X-ray image..."):
            img = preprocess_image(image)
            prob = model.predict(img)[0][0]

            prediction = "Pneumonia" if prob > 0.5 else "Normal"
            diagnosis, risk, advice = doctor_final_report(prob)

        st.success("Analysis Completed ✅")

        st.markdown("### 🧪 AI Prediction")
        st.write(f"**Prediction:** {prediction}")
        st.write(f"**Pneumonia Probability:** {prob*100:.2f}%")

        st.markdown("### 🧑‍⚕️ Doctor Final Report")
        st.write(f"**Diagnosis:** {diagnosis}")
        st.write(f"**Risk Level:** {risk}")
        st.write(f"**Doctor Advice:** {advice}")

        st.warning(
            "⚠️ This system is a decision-support tool and does not replace professional medical diagnosis."
        )
