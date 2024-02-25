import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
import pandas as pd

# Load the saved model

# from google_drive_downloader import GoogleDriveDownloader as gdd
import gdown

file_id = '1HHJ1XXN6gMOci1Y-3p5K7IX0-kFbSGJe'
url = f'https://drive.google.com/file/d/1HHJ1XXN6gMOci1Y-3p5K7IX0-kFbSGJe/uc?usp=sharing'
output = 'classifier.pkl'
import subprocess
import os
# Define the gdown command
gdown_command = ['gdown', '--id', '1HHJ1XXN6gMOci1Y-3p5K7IX0-kFbSGJe', '--output', 'classifier.pkl']

# Run the gdown command
if not os.path.exists('classifier.pkl'):
    # Run the gdown command
    subprocess.run(gdown_command, check=True)
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# with open('classifier.pkl', 'rb') as f:
#     model = joblib.load(f)
# Define the Streamlit app
def main():
    st.title('Image Upload and Prediction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img = img.resize((224, 224))
        img = np.array(img)
        image = np.expand_dims(image, axis=0)
        img = img.astype('float32')
        img = preprocess_input(img)  # Apply preprocessing
        img = img.reshape((1,) + img.shape)  # Add batch dimension

        # Load pre-trained ResNet50 model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        test_features = base_model.predict(img)
        test_features_flat = test_features.reshape(test_features.shape[0], -1)
        prediction= model.predict(test_features_flat)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        predicted_probabilities = model.predict_proba(test_features_flat)
        st.write('Prediction:', prediction)
        data = pd.DataFrame(predicted_probabilities, columns=model.classes_)

        st.write('Probability of Prediction:', data)

if __name__ == '__main__':
    main()
