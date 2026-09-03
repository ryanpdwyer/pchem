
import streamlit as st
import base64
import pandas as pd
import numpy as np
import io
import zipfile
import tempfile
import re
from io import StringIO 
from sigfig import round

def sci_form(number, sigfigs=2):
    """Format a number in scientific notation with HTML superscript."""
    formatted = round(str(number), sigfigs=sigfigs, notation='scientific')
    print(formatted)
    base, exponent = formatted.split('E')
    return f"{base}×10<sup>{exponent}</sup>"

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



def _read_thorlabs_osa(str_rep):
    """Parse a Thorlabs OSA / FTS export.

    Format: '#Thorlabs FTS' first line, '#Key;Value' header lines, a '[Data]'
    marker, then 'x;y' rows (semicolon-separated), ending with '[EndOfFile]'.
    Column names are built from the #Type / #XAxisUnit / #YAxisUnit header keys.
    """
    header = {}
    lines = str_rep.splitlines()
    try:
        data_start = next(i for i, l in enumerate(lines) if l.strip() == '[Data]') + 1
    except StopIteration:
        raise ValueError("Thorlabs file has no [Data] section")
    for line in lines[:data_start]:
        if line.startswith('#') and ';' in line:
            key, _, val = line[1:].partition(';')
            header[key.strip()] = val.strip()
    rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('['):
            break
        rows.append(line.split(';'))
    x_unit = header.get('XAxisUnit', 'nm').replace('nm_vac', 'nm').replace('nm_air', 'nm')
    x_name = f"Wavelength ({x_unit})"
    y_type = header.get('Type', 'Signal').strip() or 'Signal'
    y_name = y_type[0].upper() + y_type[1:]
    y_unit = header.get('YAxisUnit', '').strip()
    if y_unit and y_unit.upper() != 'AU':
        y_name += f" ({y_unit})"
    data = pd.DataFrame(rows, columns=[x_name, y_name]).apply(pd.to_numeric, errors='coerce')
    return data


def _decode(f):
    raw = f.getvalue() if hasattr(f, 'getvalue') else f.read()
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='replace')
    return raw


def process_file(f, skiprows=0):
    data = None
    if f.name.endswith("csv"):
        str_rep = _decode(f)
        if str_rep.lstrip().startswith('#Thorlabs') or '[SpectrumHeader]' in str_rep[:200]:
            data = _read_thorlabs_osa(str_rep)
        else:
            data = pd.read_csv(StringIO(str_rep), skiprows=skiprows)
    elif f.name.endswith("xlsx") or f.name.endswith("xls"):
        data = pd.read_excel(f, skiprows=skiprows)
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
            header_regex = r'\[.+\]'
            groups = re.split(header_regex, str_rep)
            headings = re.findall(header_regex, str_rep)
            data_start = text.index('Wavelength (nm),Absorbance')

            data = pd.read_csv(f, skiprows=data_start-1)

        elif '>>>>>Begin Spectral Data<<<<<' in str_rep:
            raw_data = np.loadtxt(f, skiprows=13)
            data = pd.DataFrame(raw_data, columns=["Wavelength (nm)", "Absorbance"])
        elif ',' in str_rep:
            # Simple CSV format with comma separator
            data = pd.read_csv(StringIO(str_rep))
        else:
            data = pd.read_table(f)
    else:
        raise NotImplementedError(f"Data loading not supported for file {f.name}")
    return data

# def process_file_zip(f):
#     data = None
#     if f.name.endswith("zip"):
        



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
    downloaded_file = df.to_excel(towrite, index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">{label}</a>'
    st.markdown(linko, unsafe_allow_html=True)

