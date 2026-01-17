
import logging
import time
import uuid
import streamlit as st
import mirpuzzle
import combineCSV
import openai_chat
import imagemodel
import imagemodel_regression
import neuralnetwork
import combineRaman
import combineEChemZip
import combineCSVElectrochem
import thermoFirstLaw
import thermoCalorimeter
import ace
import compressibility
import kinetics
import solartronData
import thermoGas
import impedance
import entropySplit
import variational
import variational_gaussian
import plotExcel
import arrhen
import random_electrons
import zeff


@st.cache_resource
def configLog():
    logging.basicConfig(filename='debug-log.log', encoding='utf-8',
            level=logging.INFO, force=True,
                format='%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',)


configLog()

# Use session_state for persistent session ID across reruns
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

startTime = time.time_ns()/1e6
logging.info(f"Start Script - id: {st.session_state.session_id}")


st.title("Chemistry and FYS Tools")

apps = {
    "CHE 120 Kinetics": arrhen,
    "MIR Puzzle": mirpuzzle,
    "1st Law of Thermodynamics": thermoFirstLaw,
    "GPT-4o-mini Image Classifier": imagemodel,
    "Zeff": zeff,
    "GPT-3 vs ChatGPT": openai_chat,
    "Neural Network Game": neuralnetwork,
    'GPT-4o-mini Image Regression': imagemodel_regression,
    "Electron Visualization": random_electrons,
"Combine UV-Vis Data": combineCSV, 
        "Combine Raman Data": combineRaman,
        "Combine CSV Electrochem": combineCSVElectrochem,
        'Electrochemistry Peak Picking': combineEChemZip,
        'Plot Excel Data': plotExcel,
        "Plot Solartron Data": solartronData,
        "Kinetics - Ocean Optics": kinetics,
        "2nd Law Calorimeter": thermoCalorimeter,
        "Mystery Gas": thermoGas,
        "Where is Equilibrium?": entropySplit,
        "Sympy Shell": ace,
        "Compressibility Factor": compressibility,
        "Variational Gaussian": variational_gaussian,
        "Linear Variational": variational,
        "Impedance": impedance
}

app = st.selectbox("Choose an application:", list(apps.keys()))

apps[app].run()

runTime = (time.time_ns()/1e6-startTime)

logging.info(f"Run time: {runTime} ms\tApp: {app}")
