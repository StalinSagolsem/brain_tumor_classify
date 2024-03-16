import os
import requests
import tensorflow as tf

def download_model_file(url, file_path):
    if not os.path.exists(file_path):
        print("Downloading model file...")
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print("Model file downloaded successfully.")
model_file_url = 'https://drive.google.com/uc?export=download&id=19nULecqAki3x9FMhTQtajn9_KKiElsVH'
model_file_path = 'trained_brain_tumor_model.h5'
download_model_file(model_file_url, model_file_path)

def load_model(model_file_path):
    return tf.keras.models.load_model(model_file_path)

# Load the model
model = load_model(model_file_path)
print(mode.summary())
