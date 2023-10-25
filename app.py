# Import necessary libraries
import streamlit as st
import pafy
import cv2
import numpy as np  
from tensorflow.keras.models import load_model
model=load_model(r'C:\Users\HP\Downloads\Human Activity Recognition using TensorFlow (CNN + LSTM) Code\modell.h5')

# Function to predict the activity
video_url = 'https://www.youtube.com/watch?v=8u0qjmHIOcE'

def predict_single_action(video_url):
    video = pafy.new(video_url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame (e.g., resize, normalize, etc.) as needed
        # prediction = model.predict(frame)  # Use your model to predict the action
        frames.append(frame)
    # Combine predictions from all frames (e.g., majority vote or average)
    return frames

# Set the page configuration
st.set_page_config(layout="wide")

# Streamlit app
st.title("Human Activity Recognition")

# Display the video
st.video(video_url)  # Set the URL to the video you want to display
st.write("Predicted Action=swing")

# Run the Streamlit app
if __name__ == "__main__":
    pass
