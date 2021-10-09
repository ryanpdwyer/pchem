

import json_tricks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
from copy import copy
import base64
from util import write_excel



def process_file(f, file_number):
    data = None
    if f.name.endswith("json"):
        data_in = json_tricks.loads(f.getvalue().decode("utf-8"))
        data = {}
        for key, val in data_in.items():
            if 'data' in val and len(val['data']) > 0:
                df = pd.DataFrame(data=val['data'],
                columns=["Potential (V)", "Current", "col1", "col2", "Time (s)"])
                df['key'] = int(key)
                df['file'] = file_number
                df['exptName'] = val['params']['experiment']
                # df['Current (mA)'] = df['Current (A)'] * 1e3
                # df['Current (nA)'] = df['Current (A)'] * 1e9
                val['data'] = df
                data[key] = val
            else:
                print(f"No data in {key}")
    else:
        raise NotImplementedError(f"Data loading not supported for file {f.name}")
    return data


@st.cache
def process_files(files):
    return {f"F{i}": process_file(file, i) for i, file in enumerate(files)}

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

scales = {'A': 1, 'mA': 1e3, 'µA': 1e6, 'nA': 1e9}

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
    expts = None
    processing="None"
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    settings = {"processing": "None"}
    st.markdown("""## Combine Solarton Electrochemistry files

This helper will allow json files from the Solartron Potentiostat to be combined and plotted.
    """)

    files = st.file_uploader("Upload JSON files",
                accept_multiple_files=True)


    if files:
        
        st.write(files)
        
        st.markdown("Files are represented below using the abbreviations F0, F1, etc...")

        for i, file in enumerate(files):
            st.write(f"F{i}: {file.name}")
        

        data_and_params_separate = process_files(files)
        
        data_and_params = {f"{out_key}-{key}":val for out_key, out_dict in data_and_params_separate.items() 
                                            for key, val in out_dict.items()}


        keys = list(data_and_params.keys())
        
        def format_func(key):
            experiment = data_and_params[key]['params']['experiment']
            return f"{key} - {experiment}"
        
        # Right now, OCP experiments are not saved (sloppy).
        st.write(data_and_params)
        expts = st.multiselect(f"Select Experiments to Plot", keys, format_func=format_func)
        

    if expts:
    
        data_matching = {key: val for key, val in data_and_params.items() if key in expts}

        params_matching = {key: val['params'] for key, val in data_matching.items()}    

        data_matching = {key: copy(val['data']) for key, val in data_matching.items()}



        st.write("""## Labels
Use the boxes below to change the labels for each line that will go on the graph.
        """)
        labels = {expt: st.text_input(f"{expt} Label:", value=expt) for expt in expts}
        

        for key, val in data_matching.items():
            val['expt-key'] = key
            val['expt'] = labels[key]


        # if ind_fname:
        #     df = data[ind_fname[0]]
        #     cols = list(df.columns)

        st.write("## Choose columns")
        with st.form("column_chooser_and_run"):
            st.write("### Experimental Parameters")
            first_expt = expts[0]
            st.write(params_matching[first_expt])
            df = data_matching[first_expt]
            cols = list(df.columns)
            x_column = st.selectbox("Choose the x column: ", cols)
            y_column = st.selectbox("Choose y column: ", cols, index=1)

            submitted = st.form_submit_button()

        
        st.session_state.ever_submitted = submitted | st.session_state.ever_submitted

        use_plotly = st.checkbox("Use plotly?", value=True)


        data = list(data_matching.values())
        
        if data is not None and len(data) > 0:

            # data, settings = limit_x_values(data, x_column, settings)
            data, settings = scale_current(data, y_column, settings)

 

            df_all = pd.concat(data)

            if not use_plotly:
                fig, ax = plt.subplots()
            
            if use_plotly:
                facet_col = st.selectbox('Facet columns?', [None, 'expt', 'file'],
                                                key='facet_col')
                fig = px.line(df_all, x=x_column, y=y_column,
                                    line_group='expt', color='expt', 
                                    facet_col=facet_col)
            else:
                for df, expt in zip(data, expts):
                    ax.plot(df[x_column].values, df[y_column].values, label=labels[expt])
            

            y_label_default = f"{y_column} ({settings['y_scale']})"


            st.markdown("### Plotting options")    
            x_label = st.text_input("x-axis label: ", value=x_column)
            y_label = st.text_input('y-axis label: ', value=y_label_default)

            if use_plotly:
                fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                st.plotly_chart(fig)
            else:
                grid = st.checkbox("Grid?", value=False)
                if grid:
                    ax.grid()
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()

                st.pyplot(fig)
                # download_figure("Download Figure", fig, "CV")

            # # Saving
            st.markdown("### Output options")
            filename = st.text_input("Filename:", value="data")
            write_excel(df_all, filename)
            st.write(df_all)

if __name__ == "__main__":
    run()

