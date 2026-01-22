"""Combine UV-Vis Data"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.combineCSV import run
run()
