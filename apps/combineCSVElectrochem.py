
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
import base64
from combineCSV import process_file


def limit_x_values(data, x_column, settings):
    st.markdown("### Limit x Range")
    x_min = st.number_input("Choose minimum x:", value=min([min(df[x_column].values) for df in data]))
    x_max = st.number_input("Choose maximum x:", value=max([max(df[x_column].values) for df in data]))
    settings['x_min'] = x_min
    settings['x_max'] = x_max
    data_out = []
    for df in data:
        mask = (df[x_column].values > x_min) * (df[x_column].values < x_max)
        data_out.append(df[mask])
    return data_out, settings


# def process_data(data, y_column, settings):
#     st.markdown("### Rescale y-axis")
#     st.selectbox("Choose y-axis scale:", value=[0, 3, 6, 9], format_func=


def run():
    df = None
    cols = None
    x_column = y_column = None
    combined_data = None
    processing="None"
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    settings = {"processing": "None"}
    st.markdown("""## Combine CSV Electrochemistry files

This helper will combine multiple CSV files (or Excel spreadsheets)
for easy plotting.

    """)

    files = st.file_uploader("Upload CSV or Excel Files",
                accept_multiple_files=True)


    if files:
        st.write(files)

        filenames = [(i, f.name) for i, f in enumerate(files)]
        data = [process_file(f) for f in files]

        ind_fname = st.selectbox("Choose data to display: ", filenames,
            format_func=lambda x: x[1], index=0)

        st.write("""## Labels
Use the boxes below to change the labels for each line that will go on the graph.
        """)
        labels = [st.text_input(f"{filename[0]}. {filename[1]}", value=filename[1]) for filename in filenames]
        
        if ind_fname:
            df = data[ind_fname[0]]
            cols = list(df.columns)
    

        st.write("## Choose columns")
        with st.form("column_chooser_and_run"):
            x_column = st.selectbox("Choose the x column: ", cols)
            y_column = st.selectbox("Choose y column: ", cols, index=len(cols)-1)

            submitted = st.form_submit_button()

        
        st.session_state.ever_submitted = submitted | st.session_state.ever_submitted

                
        if data is not None:

            data, settings = limit_x_values(data, x_column, settings)
            # data, settings = normalize_data(data, x_column, settings)
            # x_data = combined_data[x_column].values
            # Plotting
            fig, ax = plt.subplots()
            for df, fname, label in zip(data, filenames, labels):
                ax.plot(df[x_column].values, df[y_column].values, label=str(fname[0])+"-"+label)
            

            y_label_default = ""
            if settings['processing'] != 'None':
                y_label_default += settings['processing']+" "
            y_label_default+=y_column


            st.markdown("### Plotting options")    
            x_label = st.text_input("x-axis label: ", value=x_column)
            y_label = st.text_input('y-axis label: ', value=y_label_default)
            grid = st.checkbox("Grid?", value=False)
            if grid:
                ax.grid()

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.legend()
            st.pyplot(fig)

            # # Saving
            # st.markdown("### Output options")
            # st.write(combined_data)
            # filename = st.text_input("Filename:", value="data")
            # write_excel(combined_data, filename)

if __name__ == "__main__":
    run()
