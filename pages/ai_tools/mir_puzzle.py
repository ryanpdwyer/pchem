"""MIR Puzzle"""
import streamlit as st
st.page_link("pages/home.py", label="← Home")

from pchemapps.mirpuzzle import run
run()
