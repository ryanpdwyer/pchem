"""Combine Raman Data"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.combineRaman import run
run()
