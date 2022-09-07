
import streamlit as st
import base64
import pandas as pd
import numpy as np
import io
import tempfile
import re
from io import StringIO 

# def create_file( suffix='.png'):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpfile:
#         fig.savefig(tmpfile.name, format="png", dpi=300)
#     return tmpfile


# def download_figure(label, fig, default_filename, suffix='.png'):
#     filename = st.text_input("Filename:", default_filename)
#     download_figure = st.download_button(label, filename=filename+suffix,)

def limit_x_values(data, x_column, settings, step=None):
    st.markdown("### Limit x Range")
    x_min = st.number_input("Choose minimum x:", value=min([min(df[x_column].values) for df in data]), step=step)
    x_max = st.number_input("Choose maximum x:", value=max([max(df[x_column].values) for df in data]), step=step)
    settings['x_min'] = x_min
    settings['x_max'] = x_max
    data_out = []
    for df in data:
        mask = (df[x_column].values > x_min) * (df[x_column].values < x_max)
        data_out.append(df[mask])
    return data_out, settings



def process_file(f):
    data = None
    if f.name.endswith("csv"):
        data = pd.read_csv(f)
    elif f.name.endswith("xlsx") or f.name.endswith("xls"):
        data = pd.read_excel(f)
    elif f.name.endswith("Absorbance"):
        raw_data = np.loadtxt(f, skiprows=19, max_rows=2048)
        data = pd.DataFrame(raw_data, columns=["Wavelength (nm)", "Absorbance"])
    elif f.name.endswith("Transmittance"):
        raw_data = np.loadtxt(f, skiprows=19, max_rows=2048 )
        data = pd.DataFrame(raw_data, columns=["Wavelength (nm)", "Transmittance"])
    elif f.name.endswith("txt"):
        str_rep = f.getvalue().decode("utf-8")
        if 'Instrument Name:,UV-1800' in str_rep:
            text=str_rep.splitlines()
            header_regex = '\[.+\]'
            groups = re.split(header_regex, str_rep)
            headings = re.findall(header_regex, str_rep)
            data_start = text.index('Wavelength (nm),Absorbance')

            data = pd.read_csv(f, skiprows=data_start-1)

        elif '>>>>>Begin Spectral Data<<<<<' in str_rep:
            raw_data = np.loadtxt(f, skiprows=13)
            data = pd.DataFrame(raw_data, columns=["Wavelength (nm)", "Absorbance"])
        else:
            data = pd.read_table(f)
    else:
        raise NotImplementedError(f"Data loading not supported for file {f.name}")
    return data


class Enlighten_Data:
    
    def __init__(self, f):
        fh = StringIO(f.getvalue().decode("utf-8"))
        data = fh.readlines()
        header = {}
        for line in data[:33]:
            keyval = str(line).split(',', maxsplit=1)
            key=keyval[0]
            if len(keyval)==1:
                val = ""
                
            else:
                val = keyval[1]
            header[key] = val.strip()
        

        fh.seek(0)
        df = pd.read_csv(fh, skiprows=34)
        df['Label'] = header['Label']
        df['Laser Power'] = float(header['Laser Power'])
        df['Integration Time'] = float(header["Integration Time"])
        df['Scan Averaging'] = float(header['Scan Averaging'])
        df['Reprocessed'] = df['Processed'] / (df['Laser Power'] * df['Integration Time']) 
            
        important = ['Measurement ID', "Label", "Integration Time", "Timestamp", 'Laser Power', "Scan Averaging"]
        
        self.important = {key:val for key, val in header.items() if key in important}
        self.header = header
        self.df = df





def process_raman(f):
    if f.name.endswith("csv"):
        return Enlighten_Data(f)
    else:
        raise NotImplementedError(f"Data loading not supported for file {f.name}")



def find(val, array):
    return np.argmin(abs(array - val))


def write_excel(df, filename, label="Download Excel file"):
    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">{label}</a>'
    st.markdown(linko, unsafe_allow_html=True)