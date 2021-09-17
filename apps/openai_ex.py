import os
import openai
import textwrap
from copy import copy
import streamlit as st


openai.api_key = os.getenv("OPENAI_API_KEY")

def run():
    st.markdown("""# AI Playground
Try out the OpenAI model GPT-3 below.
    """)

    if 'settings' not in st.session_state:
        st.session_state.settings = {"AI Chatbot" : dict(
            engine="davinci",
             temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=["\n", " Human:", " AI:"],
prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: \nAI:",
                                        ),
                "Python bug fixer / code writer": dict(engine="davinci-codex",
  prompt="##### Fix bugs in the below function\n \n### Buggy Python\nimport Random\na = random.randint(1,12)\nb = random.randint(1,12)\nfor i in range(10):\n    question = \"What is \"+a+\" x \"+b+\"? \"\n    answer = input(question)\n    if answer = a*b\n        print (Well done!)\n    else:\n        print(\"No.\")\n    \n### Fixed Python",
  temperature=0,
  max_tokens=182,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["###"])
  }



    mode = st.selectbox("Mode:", list(st.session_state.settings.keys()))

    with st.form(key="my_form"):
        prompt = st.text_area(label="Prompt", height=400,
    value=st.session_state.settings[mode]["prompt"])
        submit_button = st.form_submit_button()


    if submit_button:
        engine = st.session_state.settings[mode]["engine"]
        kwargs = copy(st.session_state.settings[mode])
        kwargs.pop("engine")
        kwargs.pop("prompt")
        response = openai.Completion.create(engine=engine,
        prompt=prompt,
        **kwargs)
        if mode != "AI Chatbot":
            st.markdown("Response:\n\n"+textwrap.indent(response.choices[0].text, "    "))
        else:
            st.write(response.choices[0].text)



if __name__ == "__main__":
    run()
