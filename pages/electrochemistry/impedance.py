"""Impedance Analysis"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.impedance import run
run()
