import streamlit as st
from streamlit_ace import st_ace



def run():
    # Spawn a new Ace editor
    content = st_ace(language='python')
    # Display editor's content as you type
    st.write(content)


if __name__ == '__main__':
    run()