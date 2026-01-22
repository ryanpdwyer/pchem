"""Combine Electrochem CSV Data"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.combineCSVElectrochem import run
run()
