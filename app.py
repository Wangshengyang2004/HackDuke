import streamlit as st
from PIL import Image
from utils import segment_single_image, segment_images_to_video
import matplotlib.pyplot as plt
from natsort import natsorted
import glob
import cv2
import os
import numpy as np
import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
# ----------------------- Some functions -----------------------------
from googletrans import Translator

def translate_chinese_to_english(chinese_text):
    '''Example usage:
    # chinese_sentence = "你好，世界"  # Replace with your Chinese sentence.
    # translated_sentence = translate_chinese_to_english(chinese_sentence)
    # print("Chinese: ", chinese_sentence)
    # print("English: ", translated_sentence)'''
    translator = Translator()
    
    # Translate the Chinese text into English.
    translated = translator.translate(chinese_text, src='zh-CN', dest='en')
    
    # Return the translated text.
    return translated.text



# Set fixed settings (equivalent to parsed arguments in the original code)
CHECKPOINT_PATH = '/mnt/h/HackDuke/Models/hackduke_llm'
CPU_ONLY = False


@st.cache_resource
def init_model():
    device_map = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        device_map=device_map,
        trust_remote_code=True
    ).eval()

    config = GenerationConfig.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH,
        trust_remote_code=True
    )
    return model, tokenizer

def clear_chat_history():
    del st.session_state.messages

def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
        st.write(f"{avatar} {message['role']}: {message['content']}")

    return st.session_state.messages


# ----------------------- Main page ----------------------------------
st.set_page_config(page_title='COVID-19 Chatbot', page_icon=':microbe:', layout='wide')
st.title('COVID-19 Chatbot')
st.markdown('''
This is a chatbot that answers questions about COVID-19. It is based on the [COVID-QA]
            ''')
option = st.selectbox('Please select a function',('U-Net Segmentation', 'Chatbot', 'About'))


if option == 'U-Net Segmentation':
    tab1, tab2, tab3 = st.tabs(["Image Segmentation", "Video Segmentation", "About"])
    with tab1:
        st.header('Segmentation')
        
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            st.write("Uploaded image:")
            pil_image = Image.open(uploaded_file)
            st.image(pil_image, caption='Uploaded Image', use_column_width=True)
            if st.button('Inference'):    
                st.write("Performing segmentation...")
                segmented_image = segment_single_image(pil_image)
            
                st.write("Segmented image:")
                st.image(segmented_image, caption='Segmented Image', use_column_width=True)

    with tab2:
        st.header('Video Segmentation')
        on = st.toggle('Using example data', value=False)
        if on:
            # Assuming test images are stored in 'processed/test' directory
            test_img_paths = natsorted(glob.glob('processed/test/*/img_*'))
            test_images = [Image.open(img_path).convert('L') for img_path in test_img_paths]
            
            # Generate the video
            segment_images_to_video(test_images, 'example_segmented_video.mp4')
            
            st.write('Video generated: example_segmented_video.mp4')


    with tab3:
        st.header('About')
        st.write('''### This project aims to build a chatbot that can answer questions about COVID-19, and can also segment the lung from the CT scan image.
                #### This is a project for HackDuke 2023 September Health Track.
                #### The chatbot is fine-tuned from Qwen-7B-Chat model from Aliyun.
                ''')
        st.write("Team member: Shengyang Wang, Guangzhi Su")
        st.write('**Github**: github.com/Wangshengyang2004')
        st.write('**Devpost**: devpost.com/software/covid-19-cv-chatbot')
        st.write('Reference: https://github.com/hiyouga/LLaMA-Efficient-Tuning, https://github.com/QwenLM/Qwen-7B/tree/main')


if option == 'Chatbot':
    st.header('Chatbot')
    st.write('chatbot, start your conversation')
    model, tokenizer = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            for response in model.chat(tokenizer, messages, stream=True):
                placeholder.markdown(response)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response + translate_chinese_to_english(response)})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

    st.button("清空对话", on_click=clear_chat_history)
