
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
import base64
import re
from collections import defaultdict
from util import find, write_excel, process_file
import util



def combine_spectra(dataframes, labels, xcol, ycol, tol=1e-6):
    x_data = dataframes[0][xcol].values
    all_data = [x_data]
    col_names = [xcol]
    col_names.extend(labels)
    for df in dataframes:
        x = df[xcol].values
        if (len(x) != len(x_data)) or abs(x - x_data).max() > tol:
            st.write("X axes are different - Try deselecting `Same x axis?` and Submit again.")
            raise ValueError("X axis of each dataset should be the same!")
        
        y = df[ycol].values
        all_data.append(y)
        # ind, fname = ind_fname
        # before_ext = fname.split(".")[0]
        # col_names.append(f"{ind}-{before_ext}")
    
    return pd.DataFrame(np.array(all_data).T, columns=col_names)


def limit_x_values(combined_data, x_column, settings):
    st.markdown("### Limit x Range")
    x_data = combined_data[x_column].values
    x_min_val = st.selectbox("Choose minimum x:", x_data, index=0 )
    i_min = find(x_min_val, x_data)
    x_max_val = st.selectbox("Choose maximum x:", x_data, index=len(x_data)-1 )
    i_max = find(x_max_val, x_data)
    combined_data = combined_data.iloc[i_min:i_max, :]
    settings['x_min'] = x_min_val
    settings['x_max'] = x_max_val
    return combined_data, settings





def normalize_data(combined_data, x_column, settings):
    st.markdown("### Normalization options")
    x_data = combined_data[x_column].values
    processing_options = ['None', "Normalized", "Relative"]
    processing = st.selectbox("Processing?", processing_options)
    settings['processing'] = processing
    if processing == 'Normalized':
        normalize_wavelength = st.selectbox("Normalize data at: ", x_data)
        settings['normalization_wavelength'] = normalize_wavelength
    else:
        settings.pop('normalization_wavelength', 0)

    if processing == "Normalized":
        norm_ind = find(normalize_wavelength, x_data)
        y_data = combined_data.values[:, 1:]
        combined_data.values[:, 1:] = y_data / y_data[norm_ind]
                
    if processing == "Relative":
        # Should probably be tweaked a bit to be more convenient...
        y_data = combined_data.values[:, 1:]
        combined_data.values[:, 1:] = y_data / y_data.max(axis=0)
    
    return combined_data, settings

def check_nans(df, col, threshold=0.5):
    return df[col].isna().sum() / len(df) > threshold

@st.cache
def sort_files_and_create_data(files, sort_files):
    if sort_files:
        files = sorted(files, key=lambda x: x.name.split('__')[-1])
    else:
        files = files
    filenames = [(i, f.name) for i, f in enumerate(files)]
    data = [process_file(f) for f in files]
    return filenames, data

@st.cache
def create_data_dict(filenames, data):
    files_dict = defaultdict(lambda : dict(times=[], data=[], number=[], time=[]))
    # df_all = pd.DataFrame()
    for filename, d in zip(filenames, data):
        dataname, number, time = filename[1].split('__')
        dataname_short = dataname.strip('_Absorbance')
        hr, min_, sec, msec = time.split('-')
        msec = msec.split('.')[0]
        time = int(hr) * 3600 + int(min_)*60 + int(sec) + int(msec)/1000.0

        dict_entry = files_dict[dataname_short]
        dict_entry['times'].append(time)
        dict_entry['data'].append(d)
        dict_entry['number'].append(number)
        dict_entry['time'].append(f'{hr}:{min_}:{sec}.{msec}')
    return files_dict

def run():
    df = None
    cols = None
    x_column = 'Wavelength (nm)'
    y_column = 'Absorbance'
    combined_data = None
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    settings = {}
    st.markdown("""## UV-Vis Kinetics Analysis

This helper will combine multiple UV-Vis files (from Ocean Optics Ocean View .csv export)
plot/normalize the spectra, and output a single Excel file for easy plotting and analysis.

    """)

    sort_files = st.checkbox("Sort files?", value=True)

    files = st.file_uploader("Upload CSV or Excel Files",
                accept_multiple_files=True)

    if files:
        filenames, data = sort_files_and_create_data(files, sort_files)
        files_dict = create_data_dict(filenames, data)

        st.write("""### Data Summary""")

        st.markdown(f"{len(filenames)} data files from {len(files_dict)} experiments.")

        st.write("""## Labels
Use the boxes below to change the labels for each kinetics experiment.
        """)
        labels = [st.text_input(key, value=key) for key in files_dict.keys()]
    
        same_x = False
        data_example = list(files_dict.values())[0]['data'][0]


        x_data = data_example[x_column].values # Use the first x data to set all the limits.

        wavelength_monitor = st.number_input("Monitor wavelength:", min_value=x_data.min(),
                                    max_value=x_data.max(), value=x_data.mean())

        wavelength_bandwidth = st.number_input("Bandwidth", min_value=0.5, value=3.0)

        
        # Assuming all have the same x axis data
        kinetics_mask = ((x_data > wavelength_monitor-wavelength_bandwidth/2) 
                        * (x_data < wavelength_monitor+wavelength_bandwidth/2))
        

        plot_kinetics = st.checkbox("Plot kinetics data?")

        dfs = []
        for key, val in files_dict.items():
            times = np.array(val['times'])
            times = times - times.min() # Times in seconds, relative to start of experiment 
            data = np.array([np.mean(d[y_column].values[kinetics_mask]) for d in val['data']])

            dfs.append(
                pd.DataFrame.from_dict({'Time (s)':times, 'A': data, 'name':key,
                                                'wavelength': wavelength_monitor, 
                                                'bandwidth': wavelength_bandwidth,
                                                'number': val['number'],
                                                'time': val['time']}, 
                            orient='columns')
            )
        
        df_kinetics = pd.concat(dfs, ignore_index=True)

        df_kinetics['Time (min)'] = df_kinetics['Time (s)'] / 60.0
        st.write(df_kinetics)

        if plot_kinetics:

            scatter = px.line(df_kinetics, x='Time (s)', y='A', color='name',
            labels={'A': f'A @ {wavelength_monitor:.1f}±{wavelength_bandwidth/2} nm'})
            st.plotly_chart(scatter)

            st.markdown("### Output options")
            filename = st.text_input("Filename:", value="kinetics-data")
            write_excel(df_kinetics, filename)

            


if __name__ == "__main__":
    run()
