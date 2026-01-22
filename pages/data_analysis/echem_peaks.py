"""Electrochemistry Peak Picking"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.combineEChemZip import run
run()
