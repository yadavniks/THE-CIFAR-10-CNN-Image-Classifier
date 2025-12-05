import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set page config
st.set_page_config(page_title='CIFAR-10 CNN Image Classifier', layout='wide')

# Title and description
st.title('THE CIFAR-10 CNN Image Classifier')
st.markdown(
    'Upload an image for one of the following classes: '
    '[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck].'
)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Build a simple CNN model
@st.cache_resource
def build_and_load_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Load CIFAR-10 weights from Keras (training on-the-fly can also be done)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)  # Quick training for demo
    return model

model = build_and_load_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

# Image preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((32,32))
    image_array = np.array(image)
    if image_array.shape[2] == 4:
        image_array = image_array[:,:,:3]
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Prediction
def predict_image_class(image: Image.Image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Button to predict
if st.button('Predict Class'):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        predicted_class, confidence = predict_image_class(image)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader(f"Predicted Class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please upload an image file.")
