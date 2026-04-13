import streamlit as st
import numpy as np
from PIL import Image
import random

st.title("🩺 Kidney Condition Diagnosis (AI)")

file = st.file_uploader("Upload Kidney Image", type=["jpg","png","jpeg"])

classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

if file:
img = Image.open(file).convert("RGB")
st.image(img, caption="Uploaded Image", use_column_width=True)

```
# Simulated prediction (random for now)
prediction = random.choice(classes)
confidence = random.uniform(80, 99)

st.success(f"Prediction: {prediction}")
st.write(f"Confidence: {confidence:.2f}%")

st.info("⚠️ This is a demo model (no real CNN used)")
```
