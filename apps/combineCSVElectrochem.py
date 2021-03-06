
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
import base64
from util import process_file


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

scales = {'A': 1, 'mA': 1e3, 'µA': 1e6}

def scale_current(data, y_column, settings):
    st.markdown("### Scale Current")
    scale = st.selectbox("Scale:", list(scales.keys()), index=1)
    settings['y_scale'] = scale
    data_out = []
    for df in data:
        df2 = df.copy()
        df2[y_column] = df2[y_column] * scales[scale]
        data_out.append(df2)
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

        use_plotly = st.checkbox("Use plotly?", value=False)

        if data is not None:

            data, settings = limit_x_values(data, x_column, settings)
            data, settings = scale_current(data, y_column, settings)

            # data, settings = normalize_data(data, x_column, settings)
            # x_data = combined_data[x_column].values
            # Plotting
            if use_plotly:
                fig = go.Figure()
            else:
                fig, ax = plt.subplots()
            for df, fname, label in zip(data, filenames, labels):
                if use_plotly:
                    fig.add_trace(go.Line(x=df[x_column], y=df[y_column], name=str(fname[0])+"-"+label))
                else:
                    ax.plot(df[x_column].values, df[y_column].values, label=str(fname[0])+"-"+label)
            

            y_label_default = f"{y_column} ({settings['y_scale']})"


            st.markdown("### Plotting options")    
            x_label = st.text_input("x-axis label: ", value=x_column)
            y_label = st.text_input('y-axis label: ', value=y_label_default)
            grid = st.checkbox("Grid?", value=False)

            if grid and not use_plotly:
                ax.grid()

            if use_plotly:
                fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                st.plotly_chart(fig)
            else:
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
