import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fpdf import FPDF
import sqlite3
from datetime import datetime

# -----------------------------
# Load Model
# -----------------------------
model = load_model("kidney_cnn_model.keras")

class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

# -----------------------------
# Database Setup
# -----------------------------
conn = sqlite3.connect("prediction_history.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    condition TEXT,
    confidence REAL,
    date TEXT
)
""")

conn.commit()

# -----------------------------
# Doctor Recommendation
# -----------------------------
def doctor_recommendation(condition):

    recommendations = {
        "Cyst": "Consult a Nephrologist. Most kidney cysts are benign but may require monitoring.",
        "Normal": "Kidney appears normal. Maintain healthy lifestyle and hydration.",
        "Stone": "Consult a Urologist. Drink plenty of water. Further imaging may be required.",
        "Tumor": "Immediate consultation with an Oncologist or Urologist is recommended."
    }

    return recommendations.get(condition, "Consult a medical professional.")


# -----------------------------
# PDF Report Function
# -----------------------------
def generate_pdf(name, age, gender, prediction, confidence, recommendation):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200,10,"Kidney Disease Diagnosis Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    pdf.cell(200,10,f"Patient Name: {name}", ln=True)
    pdf.cell(200,10,f"Age: {age}", ln=True)
    pdf.cell(200,10,f"Gender: {gender}", ln=True)

    pdf.ln(5)

    pdf.cell(200,10,f"Predicted Condition: {prediction}", ln=True)
    pdf.cell(200,10,f"Confidence: {confidence:.2f}%", ln=True)

    pdf.ln(5)

    pdf.multi_cell(0,10,f"Doctor Recommendation: {recommendation}")

    pdf.ln(10)

    pdf.multi_cell(
        0,10,
        "This report was generated using a Convolutional Neural Network (CNN) "
        "based Artificial Intelligence system for kidney disease detection."
    )

    file_name = "kidney_diagnosis_report.pdf"
    pdf.output(file_name)

    return file_name


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🩺 Kidney Condition Diagnosis using CNN")

st.subheader("👤 Patient Details")

patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Age", min_value=1, max_value=120)
patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.write("Upload a kidney CT image to predict the condition")

uploaded_file = st.file_uploader(
    "Choose a kidney image", type=["jpg","jpeg","png"]
)


# -----------------------------
# Prediction Pipeline
# -----------------------------
if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize
    img_resized = img.resize((128,128))

    # Convert to array
    img_array = np.array(img_resized)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100


    # Reject random images
    if confidence < 70:
        st.error("Invalid image. Please upload a kidney CT scan.")
        st.stop()


    # -----------------------------
    # Show Prediction
    # -----------------------------
    st.success(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")


    # -----------------------------
    # Doctor Recommendation
    # -----------------------------
    recommendation = doctor_recommendation(predicted_class)

    st.subheader("🩺 Doctor Recommendation")

    st.info(recommendation)


    # -----------------------------
    # Save Prediction to Database
    # -----------------------------
    cursor.execute(
        "INSERT INTO history (name, age, gender, condition, confidence, date) VALUES (?, ?, ?, ?, ?, ?)",
        (
            patient_name,
            patient_age,
            patient_gender,
            predicted_class,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()


    # -----------------------------
    # Probability Chart
    # -----------------------------
    prob_df = pd.DataFrame({
        "Condition": class_labels,
        "Probability": predictions[0]
    })

    st.subheader("🔍 Prediction Probability Distribution")

    st.bar_chart(prob_df.set_index("Condition"))


    # -----------------------------
    # Generate PDF Report
    # -----------------------------
    pdf_file = generate_pdf(
        patient_name,
        patient_age,
        patient_gender,
        predicted_class,
        confidence,
        recommendation
    )

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download Medical Report (PDF)",
            data=f,
            file_name="kidney_diagnosis_report.pdf",
            mime="application/pdf"
        )


# -----------------------------
# Show Prediction History
# -----------------------------
st.subheader("📊 Prediction History")

history_df = pd.read_sql_query("SELECT * FROM history", conn)

st.dataframe(history_df)