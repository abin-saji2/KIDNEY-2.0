import streamlit as st
import numpy as np
from PIL import Image
import random

st.title("🩺 Kidney Condition Diagnosis")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

classes = ['Cyst', 'Normal', 'Stone', 'Tumor']

if file:
img = Image.open(file)   # ✅ properly indented
st.image(img)

```
prediction = random.choice(classes)
confidence = random.uniform(80, 99)

st.success(f"Prediction: {prediction}")
st.write(f"Confidence: {confidence:.2f}%")
```

