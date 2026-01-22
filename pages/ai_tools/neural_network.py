"""Neural Network Game"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.neuralnetwork import run
run()
