import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load your pre-trained Keras model
model = load_model("your_model.h5")  # Replace with the path to your model


# Define a function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    # Resize and convert image to array format, assuming 224x224 as input size
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Streamlit app layout
st.title("Image Classification with TensorFlow and Streamlit")
st.write("Upload an image to classify using your model.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess and make predictions
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    # Display predictions
    st.write("Prediction:", predictions)
