"""Plot Solartron Data"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.solartronData import run
run()
