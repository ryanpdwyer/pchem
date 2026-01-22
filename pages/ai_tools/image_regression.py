"""GPT-4o-mini Image Regression"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.imagemodel_regression import run
run()
