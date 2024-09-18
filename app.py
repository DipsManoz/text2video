import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the CogVideoX model and tokenizer
@st.cache_resource
def load_model():
    model_name = "THUDM/CogVideoX-5b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit interface
st.title("Text to Video Generator using CogVideoX-5b")

# Input text prompt from user
prompt = st.text_input("Enter a text prompt for video generation:", "")

# Button to generate the video
if st.button("Generate Video"):
    if prompt:
        with st.spinner("Generating video..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs)
            
            # Assuming video output is a tensor; simulate video path
            video_path = "generated_video.mp4"
            with open(video_path, "wb") as f:
                f.write(output[0].cpu().numpy())  # Example write operation (modify this as per the actual model's output)

            st.video(video_path)
    else:
        st.warning("Please enter a prompt before generating the video.")

# Footer
st.write("Powered by THUDM/CogVideoX-5b and Streamlit")
