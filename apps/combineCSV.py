
import numpy as np
import pandas as pd
import streamlit as st
import io
import base64


def write_excel(df, filename, label="Download Excel file"):
    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()  # some strings
    linko= f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">{label}</a>'
    st.markdown(linko, unsafe_allow_html=True)


def process_file(f):
    data = None
    if f.name.endswith("csv"):
        data = pd.read_csv(f)
    elif f.name.endswith("xlsx"):
        data = pd.read_excel(f)
    return data

def combine_spectra(dataframes, filenames, xcol, ycol, tol=1e-6):
    x_data = dataframes[0][xcol].values
    all_data = [x_data]
    col_names = [xcol]
    for df, ind_fname in zip(dataframes, filenames):
        x = df[xcol].values
        if abs(x - x_data).max() > tol:
            raise ValueError("X axis of each dataset should be the same!")
        
        y = df[ycol].values
        all_data.append(y)
        ind, fname = ind_fname
        before_ext = fname.split(".")[0]
        col_names.append(f"{ind}-{before_ext}")
    
    return pd.DataFrame(np.array(all_data).T, columns=col_names)
def run():
    df = None
    cols = None
    x_column = y_column = None
    st.markdown("""## Combine CSV files

This helper will combine multiple CSV files (or Excel spreadsheets)
into a single Excel file for easy plotting and anaysis.

    """)

    files = st.file_uploader("Upload CSV or Excel Files",
                accept_multiple_files=True)

    st.write(files)

    filenames = [(i, f.name) for i, f in enumerate(files)]
    data = [process_file(f) for f in files]

    st.write(data)

    ind_fname = st.selectbox("Choose data to display: ", filenames,
            format_func=lambda x: x[1])

    if ind_fname:
        df = data[ind_fname[0]]
        cols = list(df.columns)
    
    with st.form("column_chooser_and_run"):

        x_column = st.selectbox("Choose the x column: ", cols)
        y_column = st.selectbox("Choose y column: ", cols)

        submitted = st.form_submit_button()
        if submitted:
            combined_data = combine_spectra(data, filenames, x_column, y_column)
            st.write(combined_data)
            filename = st.text_input("Filename:", value="data")
            write_excel(combined_data, filename)




if __name__ == "__main__":
    run()
