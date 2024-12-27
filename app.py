import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Constants
SEQUENCE_LENGTH = 16  # Number of frames per sequence
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  # Resize dimensions
CLASSES_LIST = ["Non-Violent", "Violent"]  # Class labels

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = "violence_detection_model.h5"  # Update with your model path
    return load_model(model_path)

# Function to extract frames from a video
def extract_frames(video_path, sequence_length, image_height, image_width):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)

    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame, (image_width, image_height))
        normalized_frame = resized_frame / 255.0  # Normalize frame to [0, 1]
        frames_list.append(normalized_frame)

    video_reader.release()
    return np.array(frames_list)

# Function to predict the class of a video
def predict_video_class(video_path, model):
    frames = extract_frames(video_path, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)
    if len(frames) < SEQUENCE_LENGTH:
        st.error("The video is too short. Please upload a longer video.")
        return None

    # Add batch dimension
    input_data = np.expand_dims(frames, axis=0)  # Shape: (1, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    return CLASSES_LIST[predicted_class], prediction[0]

# Streamlit App
def main():
    st.title("Violence Detection in Videos")
    st.write("Upload a video to classify it as **Non-Violent** or **Violent**.")

    # Upload video file
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

    if video_file is not None:
        # Save the uploaded video to a temporary file
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        st.video(temp_video_path)  # Display uploaded video

        # Load model
        model = load_trained_model()

        # Predict the class
        with st.spinner("Analyzing the video..."):
            predicted_class, probabilities = predict_video_class(temp_video_path, model)

        if predicted_class is not None:
            st.success(f"Prediction: **{predicted_class}**")
            st.write(f"Confidence Scores:")
            for i, class_name in enumerate(CLASSES_LIST):
                st.write(f"{class_name}: {probabilities[i]:.2f}")

        # Clean up the temporary video
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()
