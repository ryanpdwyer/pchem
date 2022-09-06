
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import io
import base64
from util import find, write_excel, process_raman



def combine_spectra(dataframes, labels, xcol, ycol, tol=1e-6):
    x_data = dataframes[0][xcol].values
    all_data = [x_data]
    col_names = [xcol]
    col_names.extend(labels)
    for df in dataframes:
        x = df[xcol].values
        if abs(x - x_data).max() > tol:
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

def run():
    df = None
    cols = None
    x_column = y_column = None
    combined_data = None
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    settings = {}
    st.markdown("""## Combine Raman files

This helper will combine multiple Raman files from the Wasatch Photonics spectrometer.

    """)

    files = st.file_uploader("Upload CSV Files",
                accept_multiple_files=True)


    if files:
        st.write(files)

        filenames = [(i, f.name) for i, f in enumerate(files)]
        enlighen_data = [process_raman(f) for f in files]
        data = [x.df for x in enlighen_data]

        ind_fname = st.selectbox("Choose data to display: ", filenames,
            format_func=lambda x: x[1], index=0)

        st.write("""## Labels
Use the boxes below to change the labels for each line that will go on the graph.
        """)
        labels = [st.text_input(f"{filename[0]}. {filename[1]}", value=str(filename[0])+"-"+filename[1]) for filename in filenames]

        if ind_fname:
            df = data[ind_fname[0]]
            cols = list(df.columns)
    

        st.write("## Choose columns")
        with st.form("column_chooser_and_run"):
            x_column = st.selectbox("Choose the x column: ", cols, index=2)
            y_column = st.selectbox("Choose y column: ", cols, index=len(cols)-1)
            
            same_x = st.checkbox("Same x axis?", value=True)

            submitted = st.form_submit_button()

        
        st.session_state.ever_submitted = submitted | st.session_state.ever_submitted
        
        if st.session_state.ever_submitted:

            if check_nans(data[0], x_column):
                st.markdown(f"Column `{x_column}` seems to be missing data; try selecting another column for the x-axis and **Submit** again.")
                st.session_state.ever_submitted = False
            elif check_nans(data[0], y_column):
                st.markdown(f"Column `{y_column}` seems to be missing data; try selecting another column for the y-axis and **Submit** again.")
                st.session_state.ever_submitted = False
            else:
                combined_data = combine_spectra(data, labels, x_column, y_column, same_x)

        use_plotly = st.checkbox("Use plotly?", value=True)

        if combined_data is not None:
            combined_data, settings = limit_x_values(combined_data, x_column, settings)
            combined_data, settings = normalize_data(combined_data, x_column, settings)
            x_data = combined_data[x_column].values

            y_label_default = ""
            if settings['processing'] != 'None':
                y_label_default += settings['processing']+" "
            y_label_default+=y_column


            st.markdown("### Plotting options")    
            x_label = st.text_input("x-axis label: ", value=x_column)
            y_label = st.text_input('y-axis label: ', value=y_label_default)

            # Plotting
            if use_plotly:
                plotly_fig = px.line(combined_data, x=x_column, y=combined_data.columns[1:],
                        labels={'value': y_label, x_column: x_label})
                st.plotly_chart(plotly_fig)
            else:
                fig, ax = plt.subplots()
                for col, fname, label in zip(combined_data.values[:, 1:].T, filenames, labels):
                    ax.plot(x_data, col, label=label)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()
                st.pyplot(fig)
            

            


            # Saving
            st.markdown("### Output options")
            st.write(combined_data)
            filename = st.text_input("Filename:", value="data")
            write_excel(combined_data, filename)

if __name__ == "__main__":
    run()
