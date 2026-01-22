"""GPT-3 vs ChatGPT Comparison"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.openai_chat import run
run()
