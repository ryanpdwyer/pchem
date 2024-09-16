
import streamlit as st
from PIL import Image, ImageDraw
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

    def get_response(system, image, model="gpt-4o-mini"):
            # Encode the image as a base64 string
        base64_image = encode_image(image)

        return client.chat.completions.create(
    model=model,
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

    st.title("Image regression(?) using GPT-4o-mini")

    st.write("Use the system prompt below to try to build a zero-shot regression model that places a circle over the image at a given point. Right now, it tries to find the position of the person's nose. Try modifying the prompt to do something else too.")

    if 'system' not in st.session_state:
        st.session_state.system_reg = """Respond to an input image with the relative position of the tip of the person in the image's nose. Use the format (x, y) for your output, where x = 0 is the image's left edge, x = 100 is the image's right edge, y = 0 is the image's top edge, and y = 100 is the image's bottom edge. If there is no nose in the image, respond no nose. Do not give any other response."""
    
    if 'cost' not in st.session_state:
        st.session_state.cost = 0.0
    
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-4o-mini"

    system = st.text_area("System prompt", st.session_state.system_reg)

    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)

    submit = st.button("Submit")

    # Add a webcam stream to the app with a button to take a snapshot
    picture = st.camera_input("Take a picture")

    # Resize the image to a standard size (448x252) using PIL
    if picture is not None:
        img = Image.open(io.BytesIO(picture.read()))
        x_dim = 448
        y_dim = 252
        img = img.resize((x_dim, y_dim))


        response = get_response(system, img, model)
        
        try:
            # Just get the part of the response in parentheses (x, y) - save as coords
            coords_str = response.choices[0].message.content.split('(')[1].split(')')[0]
            coords = tuple(map(float, coords_str.split(',')))
        except:
            coords = tuple()

        if len(coords) == 2:
            # Draw a circle on the image at the coordinates
            draw = ImageDraw.Draw(img)
            draw.ellipse([coords[0]*x_dim/100.0-10, coords[1]*y_dim/100.0-10, coords[0]*x_dim/100.0+10, coords[1]*y_dim/100.0+10], outline='red', width=2)
            st.image(img)

        if system != st.session_state.system_reg:
            st.session_state.system_reg = system
        
        if model != st.session_state.model:
            st.session_state.model = model



    if submit and picture is not None and (system != st.session_state.system_reg or model != st.session_state.model):
        response = get_response(system, img, model)

        st.session_state.system_reg = system
        st.session_state.model = model


    
    # Add a simple cost estimate...
        
    if response:
        st.write("Response:")
        st.write(response.choices[0].message.content)

        model_cost = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (5, 15)}

        st.write("Total Cost estimate:")
        st.session_state.cost += (response.usage.prompt_tokens*model_cost[model][0] + response.usage.completion_tokens*model_cost[model][1]) / 1e4

        st.write(f"{st.session_state.cost:.4f} cents")


if __name__ == "__main__":
    run()

