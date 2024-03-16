import requests
import tensorflow as tf

file_url = "https://drive.usercontent.google.com/download?id=19nULecqAki3x9FMhTQtajn9_KKiElsVH&export=download&confirm=t&uuid=e1436ddd-cf5a-4108-a1d5-bb3403fb2a23"
response = requests.get(file_url)

# Save the file or process the content
with open("trained_brain_tumor_model.h5", "wb") as file:
    file.write(response.content)


import time

import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu

# Function to load the model
def load_model():
    return tf.keras.models.load_model("trained_brain_tumor_model.h5")


# Function to preprocess and predict the image
def model_prediction(model, test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))
    input_array = tf.keras.preprocessing.image.img_to_array(image) / 255
    input_array = np.array([input_array])
    predictions = model.predict(input_array)[0][0]

    # Return the rounded prediction
    return predictions


# Sidebar
with st.sidebar:
    app_mode = option_menu("Dashboard", ['Home', 'About Project', 'Prediction'],
                           icons=['house', 'filter-square', 'search'],
                           default_index=0)


# Main Page
if app_mode == "Home":
    st.markdown("""
        # BrainCheck: Tumor or Healthy Classifier
        """)
    # image_path = "image-2000x380.jpg"
    # st.image(image_path,use_column_width=True)
    # Main Page
    if app_mode == "Home":
        st.markdown("""

            Welcome to BrainCheck, a powerful application designed to classify MRI scans into two distinct categories: Brain Tumor or Healthy Brain. Our cutting-edge technology utilizes a deep learning model, enabling accurate predictions based on uploaded MRI images.
            
            ---

            ## Introduction

            The ability to diagnose brain conditions efficiently is crucial for timely and effective treatment. BrainCheck aims to streamline this process by providing a user-friendly platform for classifying MRI scans. Whether you're a healthcare professional or an individual concerned about brain health, BrainCheck offers a reliable solution.

            ### Key Features

            - **Accurate Predictions:** Our model has been trained on a diverse dataset to ensure high accuracy in distinguishing between brain tumors and healthy brains.
            - **Fast Analysis:** Receive results in seconds, allowing for quick decision-making and prompt medical intervention when necessary.
            - **User-Friendly Interface:** The application is designed with simplicity in mind, making it accessible to users with varying levels of technical expertise.
            
            ---

            ## How It Works

            - **Upload Image:** Navigate to the "Prediction" section, upload an MRI scan, and let our system analyze the image.
            - **Analysis:** Our advanced algorithms process the uploaded image, identifying potential signs of a brain tumor or confirming a healthy brain.
            - **Results:** View the prediction results along with any additional information to assist in making informed decisions.
            
            ---

            ## Get Started

            Experience the power of BrainCheck by visiting the **Prediction** page in the sidebar. Upload an image and join us in the mission to improve brain health!
            
            ---

            ### Important Note

            Please ensure that the uploaded image is a valid MRI scan for accurate predictions.
        """)


# About Project
elif app_mode == "About Project":
    st.markdown("""     # ABOUT PROJECT     """)
    st.markdown("""
        ## About Dataset
        _This_ _dataset_ _contains_ _4600_ _images_ _of_ _human_ _brain_ _MRI_ _images_ _which_ _are_ _classified_ _into_ _2_ _classes:_ _'Brain_ _Tumor'_ _and_ _'Healthy'_.

        _2513_ _images_ _of_ _the_ _total_ _MRI_ _images_ _belongs_ _to_ _'Brain_ _Tumor'_ _class,_ _and_ _the_ _remaining_ _2087_ _belongs_ _to_ _'Healthy'_ _class_.

        _The_ _dataset_ _contains_ _3_ _folders:_ _'train',_ _'validation'_ _and_ _'test'_
    """)
    st.write("_The_ _dataset_ _can_ _be_ _found_ _[**here**](https://www.dropbox.com/s/jztol5j7hvm2w96/brain_tumor%20data%20set.zip)_")

# Prediction
elif app_mode == "Prediction":

    st.markdown("""     # PREDICTION     """)
    # st.header("PREDICTION")
    test_image = st.file_uploader("Please upload the MRI Scan:")

    # Load the model
    model = load_model()

    # Display the selected image
    if test_image is not None:
        st.image(test_image, width=400, use_column_width=True, caption="Selected Image")

    # Predict button
    if st.button("Predict"):
        if test_image is not None:

            # Progress bar
            progress_bar = st.progress(0)
            for perc_completed in range(100):
                time.sleep(0.02)
                progress_bar.progress(perc_completed + 1)

            # Predict and display result
            result = model_prediction(model, test_image)
            print(f"{result:.12f}")
            result = round(result)
            result_message = "Brain Tumor" if result == 0 else "Healthy Brain"

            st.success(f"Prediction Result: {result_message}")
        else:
            st.warning("Please upload an image before pressing the 'Predict' button.")
