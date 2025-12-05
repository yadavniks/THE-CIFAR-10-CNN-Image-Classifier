import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# Download model if not exists
MODEL_PATH = "cifar10_cnn_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # replace with your file ID

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# CIFAR-10 classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit UI
st.title("THE CIFAR-10 CNN Image Classifier")
st.markdown("Upload an image and the CNN will predict its class.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

def preprocess_image(image: Image.Image):
    image = image.resize((32,32))
    image_array = np.array(image)
    if image_array.shape[2] == 4:  # RGBA -> RGB
        image_array = image_array[:,:,:3]
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image_class(image: Image.Image):
    processed = preprocess_image(image)
    preds = model.predict(processed)
    class_name = class_names[np.argmax(preds)]
    confidence = np.max(preds)
    return class_name, confidence

if st.button("Predict Class"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        class_name, confidence = predict_image_class(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader(f"Predicted Class: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please upload an image file.")
