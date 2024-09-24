import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os



def run():
    # Load environment variables from .env file
    load_dotenv()

    # Initialize the OpenAI client with API key from .env
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    st.title("GPT-3 vs GPT-3.5 Chatbot Comparison")

    # Function to generate response from GPT model
    def get_gpt_response(messages, model):
        if model == "davinci-002":
            prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
            prompt += "\nAssistant:"
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=150,
                stop=["\n"]
            )
            return response.choices[0].text.strip()
        elif model == "gpt-3.5-turbo-instruct":
            prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
            response = client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=150
            )
            return response.choices[0].message.content

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.left_messages = []
        st.session_state.right_messages = []

    # Sidebar for initial prompt, model selection, and initial message button
    with st.sidebar:
        initial_prompt = st.text_area("Enter the initial prompt for both chats:")
        right_model = st.selectbox(
            "Select model for right chat",
            ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct", "gpt-3.5-turbo"]
        )
        
        if st.button("Send Initial Message", disabled=st.session_state.initialized):
            if initial_prompt:
                st.session_state.left_messages.append({"role": "user", "content": initial_prompt})
                st.session_state.right_messages.append({"role": "user", "content": initial_prompt})
                
                # Generate responses for both chats
                left_response = get_gpt_response(st.session_state.left_messages, "davinci-002")
                right_response = get_gpt_response(st.session_state.right_messages, right_model)
                
                st.session_state.left_messages.append({"role": "assistant", "content": left_response})
                st.session_state.right_messages.append({"role": "assistant", "content": right_response})
                
                st.session_state.initialized = True
                st.rerun()
            else:
                st.warning("Please enter an initial prompt before sending.")

    # Create two columns for side-by-side chat display
    left_column, right_column = st.columns(2)

    # Left chat (GPT-3: davinci-002)
    with left_column:
        st.subheader("Left Chat (GPT-3: davinci-002)")
        for message in st.session_state.left_messages:
            st.write(f"{message['role'].capitalize()}: {message['content']}")
        
        left_input = st.text_input("You (Left):", key="left_input", disabled=not st.session_state.initialized)
        if st.button("Submit Left", disabled=not st.session_state.initialized):
            st.session_state.left_messages.append({"role": "user", "content": left_input})
            response = get_gpt_response(st.session_state.left_messages, "davinci-002")
            st.session_state.left_messages.append({"role": "assistant", "content": response})
            st.rerun()

    # Right chat (selectable GPT-3.5 model)
    with right_column:
        st.subheader(f"Right Chat (GPT-3.5: {right_model})")
        for message in st.session_state.right_messages:
            st.write(f"{message['role'].capitalize()}: {message['content']}")
        
        right_input = st.text_input("You (Right):", key="right_input", disabled=not st.session_state.initialized)
        if st.button("Submit Right", disabled=not st.session_state.initialized):
            st.session_state.right_messages.append({"role": "user", "content": right_input})
            response = get_gpt_response(st.session_state.right_messages, right_model)
            st.session_state.right_messages.append({"role": "assistant", "content": response})
            st.rerun()


if __name__ == "__main__":
    run()