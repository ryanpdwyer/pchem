
import streamlit as st
from PIL import Image
import numpy as np
import io
import base64
from openai import OpenAI
import base64
from io import BytesIO
from dotenv import load_dotenv
import os

def run():
    vars = load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def encode_image(image):
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    client = OpenAI(api_key=OPENAI_API_KEY)

    def get_response(system, image):
            # Encode the image as a base64 string
        base64_image = encode_image(image)

        return client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": system
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    })

    response = None

    st.title("Image classifier using GPT-4o-mini")

    st.write("Use the system prompt below to try zero-shot image classification. Try modifying the prompt to do something else too.")

    if 'system' not in st.session_state:
        st.session_state.system = """Respond to an input image with either 'phone' or 'no phone' depending on whether the image contains a phone. Do not give any other response."""
    
    if 'cost' not in st.session_state:
        st.session_state.cost = 0.0

    system = st.text_area("System prompt", """Respond to an input image with either 'phone' or 'no phone' depending on whether the image contains a phone. Do not give any other response.""")

    submit = st.button("Submit")

    # Add a webcam stream to the app with a button to take a snapshot
    picture = st.camera_input("Take a picture")

    # Resize the image to a standard size (448x252) using PIL
    if picture is not None:
        img = Image.open(io.BytesIO(picture.read()))
        img = img.resize((448, 252))


        response = get_response(system, img)

        if system != st.session_state.system:
            st.session_state.system = system



    if submit and picture is not None and system != st.session_state.system:
        response = get_response(system, img)

        st.session_state.system = system


    
    # Add a simple cost estimate - input tokens cost $0.15 per million tokens, output tokens cost $0.60 per million tokens
        
    if response:
        st.write("Response:")
        st.write(response.choices[0].message.content)

        st.write("Total Cost estimate:")
        st.session_state.cost += (response.usage.prompt_tokens*0.15 + response.usage.completion_tokens*0.6) / 1e4

        st.write(f"{st.session_state.cost:.4f} cents")


if __name__ == "__main__":
    run()

