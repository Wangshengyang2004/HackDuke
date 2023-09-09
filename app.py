import json
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set fixed settings (equivalent to parsed arguments in the original code)
CHECKPOINT_PATH = 'Qwen/Qwen-7B-Chat'
CPU_ONLY = False

st.set_page_config(page_title="Qwen-7B-Chat")
st.title("Qwen-7B-Chat")

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

def main():
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

        st.write(f"🤖 assistant: {response}")
        messages.append({"role": "assistant", "content": response})

        st.session_state.messages = messages

    if st.button("Clear Chat"):
        clear_chat_history()
        st.session_state.messages = []

if __name__ == "__main__":
    main()
