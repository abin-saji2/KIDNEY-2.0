from tensorflow.keras.models import load_model

# Load old model

model = load_model("kidney_cnn_model.h5", compile=False)

# Save as new format

model.save("kidney_model.keras")

print("✅ Model converted successfully!")
