import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

# Load the saved model
saved_model = load_model(r"C:\Users\thiru\Documents\Final Project\IMC\IMC_saved_model.h5")

# Function to predict image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Scale pixel values to [0, 1]
    result = saved_model.predict(img_array)
    if result[0][0] >= 0.5:
        return 'Dog'
    else:
        return 'Cat'

# Streamlit app
st.title('Cat-Dog Image Classifier')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image_path = "uploaded_image.jpg"  # Save the uploaded image temporarily
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    prediction = predict_image(image_path)
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("Prediction: ", prediction)
