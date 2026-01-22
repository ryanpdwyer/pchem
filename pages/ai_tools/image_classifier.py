"""GPT-4o-mini Image Classifier"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.imagemodel import run
run()
