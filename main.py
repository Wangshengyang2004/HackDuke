import streamlit as st
from PIL import Image
from utils import segment_single_image
import matplotlib.pyplot as plt
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

import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set fixed settings (equivalent to parsed arguments in the original code)
CHECKPOINT_PATH = './Models/hackduke1_dpo'
CPU_ONLY = False

st.set_page_config(page_title="HackDuke-COVID-19-Chatbot")
st.title("COVID-19 Chatbot")

@st.cache(allow_output_mutation=True)
def init_model():
    device_map = "cpu" if CPU_ONLY else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH,
        device_map=device_map,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_PATH,
        use_fast=False,
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

tab1, tab2, tab3 = st.tabs(["Segmentation", "Chatbot", "About"])

with tab1:
    st.header('Segmentation')
    st.write('upload image')
    # Example usage:
    pil_image = Image.open('path/to/your/image.jpg')
    segmented_image = segment_single_image(pil_image)

    # Display the segmented image
    plt.imshow(segmented_image, cmap='gray')
    plt.show()


with tab2:
    st.header('Chatbot')
    st.write('chatbot, start your conversation')
    model, tokenizer = init_model()
    messages = init_chat_history()

    prompt = st.text_input("Type your message and press Enter:")

    if st.button("Send"):
        st.write(f"🧑‍💻 user: {prompt}")
        messages.append({"role": "user", "content": prompt})

        # Generate a response from the model (simplified for demonstration)
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response + translate_chinese_to_english(response)
        st.write(f"🤖 assistant: {response}")
        messages.append({"role": "assistant", "content": response})

        st.session_state.messages = messages

    if st.button("Clear Chat"):
        clear_chat_history()
        st.session_state.messages = []

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
