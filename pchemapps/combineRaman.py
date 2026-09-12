
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import io
from scipy import signal
import base64
from pchemapps.util import find, write_excel, process_raman
import pchemapps.util as util
from pchemapps.notebook_export import add_notebook_download_buttons, add_echem_notebook_download_buttons



def combine_spectra(dataframes, labels, xcol, ycol, tol=1e-3):
    x_data = dataframes[0][xcol].values
    all_data = [x_data]
    col_names = [xcol]
    col_names.extend(labels)
    for df in dataframes:
        x = df[xcol].values
        if len(x) != len(x_data) or abs(x - x_data).max() > tol:
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
    combined_data = combined_data.iloc[i_min:i_max + 1, :]
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
        combined_data = combined_data.copy()
        y_data = combined_data.iloc[:, 1:].values
        combined_data.iloc[:, 1:] = y_data / y_data[norm_ind]
                
    if processing == "Relative":
        # Should probably be tweaked a bit to be more convenient...
        combined_data = combined_data.copy()
        y_data = combined_data.iloc[:, 1:].values
        combined_data.iloc[:, 1:] = y_data / y_data.max(axis=0)
    
    return combined_data, settings

def check_nans(df, col, threshold=0.5):
    return df[col].isna().sum() / len(df) > threshold


def derivative(data, y_column, x_column, settings):

    st.markdown("""### Peak picking settings
The peak smoothing parameter can be adjusted to minimize false positives and help the peak picking algorithm find a single peak.
    """)
    dV_peak = st.number_input("Peak smoothing (x-axis units)", value=2.0)
    settings['dx_peak'] = dV_peak


    for df in data:
        dV = np.mean(np.abs(np.gradient(df[x_column].values)))
        window = signal.get_window('triang', np.round(dV_peak/dV).astype(int))
        y = df[y_column].values
        df['y_norm'] = y_norm =  y/abs(y).max()
        dy = signal.convolve(np.gradient(y_norm), window, mode='same')
        df['dy'] = dy

        # Zero crossings
        inds = np.where(np.diff(np.sign(dy)))[0]
        peaks = np.zeros_like(y, dtype=bool)
        peaks[inds] = True
        df["Peak"] = peaks

    

    return data, settings


def default_columns(kind, cols):
    """Default (x, y) column indices for each file type."""
    if kind == "Enlighten Raman":
        xi = cols.index("Wavenumber") if "Wavenumber" in cols else min(2, len(cols) - 1)
        yi = cols.index("Reprocessed") if "Reprocessed" in cols else len(cols) - 1
        return xi, yi
    return 0, min(1, len(cols) - 1)


def standardize(df, x_index, y_index, x_name, y_name):
    """Pull chosen x/y positions into a numeric frame with common names.

    Positions are used because equivalent file types can use different source
    column labels (for example, OMNIC SPA and CSV files).
    """
    out = pd.DataFrame({
        x_name: pd.to_numeric(df.iloc[:, x_index], errors="coerce").values,
        y_name: pd.to_numeric(df.iloc[:, y_index], errors="coerce").values,
    })
    return out


def run():
    x_column = y_column = None
    combined_data = None
    use_separate_x = False
    if 'ever_submitted' not in st.session_state:
        st.session_state.ever_submitted = False
    settings = {}
    st.markdown("""## Combine Raman / IR files

Combine and overlay spectra from the Wasatch Photonics Raman spectrometer (ENLIGHTEN `.csv`)
and the Nicolet FTIR (OMNIC `.spa`, or 2-column `.csv` export). Files of different types can be
mixed; choose which columns to use for each type below.

    """)

    files = st.file_uploader("Upload ENLIGHTEN .csv, OMNIC .spa, or OMNIC .csv files",
                accept_multiple_files=True)


    if files:
        file_signature = tuple((f.name, len(f.getvalue())) for f in files)
        if st.session_state.get("raman_file_signature") != file_signature:
            st.session_state.ever_submitted = False
            st.session_state.raman_file_signature = file_signature
        filenames = [(i, f.name) for i, f in enumerate(files)]
        loaded = [process_raman(f) for f in files]
        kinds = list(dict.fromkeys(d.kind for d in loaded))

        st.table(pd.DataFrame({"File": [f.name for f in files],
                               "Type": [d.kind for d in loaded],
                               "Points": [len(d.df) for d in loaded]}))

        st.write("""## Labels
Use the boxes below to change the labels for each line that will go on the graph.
        """)
        labels = [st.text_input(f"{filename[0]}. {filename[1]}", value=str(filename[0])+"-"+filename[1]) for filename in filenames]

        st.write("## Choose columns")
        with st.form("column_chooser_and_run"):
            choices = {}
            choice_names = {}
            for kind in kinds:
                cols = list(next(d.df for d in loaded if d.kind == kind).columns)
                xi, yi = default_columns(kind, cols)
                st.markdown(f"**{kind}** files")
                indices = range(len(cols))
                x_index = st.selectbox(f"x column ({kind}):", indices, index=xi,
                                       format_func=cols.__getitem__, key=f"x_index_{kind}")
                y_index = st.selectbox(f"y column ({kind}):", indices, index=yi,
                                       format_func=cols.__getitem__, key=f"y_index_{kind}")
                choices[kind] = (x_index, y_index)
                choice_names[kind] = (cols[x_index], cols[y_index])

            first_x, first_y = choice_names[kinds[0]]
            x_column = st.text_input("Name for the x column in the output:", value=first_x)
            y_column = st.text_input("Name for the y column in the output:", value=first_y)

            same_x = st.checkbox("Same x axis?", value=len(kinds) == 1,
                                 help="Uncheck to overlay spectra with different x axes (e.g. Raman + IR).")

            submitted = st.form_submit_button()


        st.session_state.ever_submitted = submitted | st.session_state.ever_submitted

        if st.session_state.ever_submitted:
            data = [standardize(d.df, *choices[d.kind], x_column, y_column) for d in loaded]

            peak_list = None

            if st.checkbox("Pick peaks?"):

                data, settings = derivative(data, y_column, x_column, settings)

                peak_list = [df[df['Peak']].loc[:, [x_column, y_column, 'y_norm']] for df in data]

                # Allow peak list to be modified - set a threshold for height, for example...

                height_threshold = st.slider("Peak height threshold", value=0.05, min_value=0.0, max_value=0.9)

                for df in peak_list:
                    df.drop(df[df['y_norm'] < height_threshold].index, inplace=True)

                st.write(peak_list)

            if any(check_nans(df, x_column) for df in data):
                st.markdown(f"The x column seems to be missing data; try selecting another column for the x-axis and **Submit** again.")
                st.session_state.ever_submitted = False
            elif any(check_nans(df, y_column) for df in data):
                st.markdown(f"The y column seems to be missing data; try selecting another column for the y-axis and **Submit** again.")
                st.session_state.ever_submitted = False
            elif same_x:
                try:
                    combined_data = combine_spectra(data, labels, x_column, y_column)
                except ValueError:
                    st.error("X axes are different - uncheck `Same x axis?` and Submit again.")
                    st.session_state.ever_submitted = False
            else:
                use_separate_x = True

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

                if peak_list is not None:
                    for df, label in zip(peak_list, labels):
                        plotly_fig.add_trace(px.scatter(df, x=x_column, y=y_column).data[0])

                st.plotly_chart(plotly_fig)
            else:
                fig, ax = plt.subplots()
                for col, fname, label in zip(combined_data.values[:, 1:].T, filenames, labels):
                    ax.plot(x_data, col, label=label)

                if peak_list is not None:
                    for df, label in zip(peak_list, labels):
                        ax.plot(df[x_column], df[y_column], 'o', label=f"{label} peaks")

                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()
                st.pyplot(fig)


            # Saving
            st.markdown("### Output options")
            st.write(combined_data)
            filename = st.text_input("Filename:", value="data")
            write_excel(combined_data, filename)

            # Notebook export
            add_notebook_download_buttons(
                combined_data=combined_data,
                settings=settings,
                labels=labels,
                x_label=x_label,
                y_label=y_label,
                x_column=x_column,
                title="Raman / IR Spectroscopy Analysis",
                data_type="Raman/IR",
                filename_base=filename
            )

        elif use_separate_x:
            data, settings = util.limit_x_values(data, x_column, settings, step=1.0)
            file_kinds = [d.kind for d in loaded]

            st.markdown("### Plotting options")
            x_label = st.text_input("x-axis label: ", value=x_column)
            stacked = False
            if len(kinds) > 1:
                stacked = st.checkbox("Stack by file type (shared x-axis, separate y-axes)?", value=True)
            if stacked:
                y_labels = {kind: st.text_input(f"y-axis label ({kind}): ", value=choice_names[kind][1], key=f"ylab_{kind}")
                            for kind in kinds}
                settings['layout'] = 'stacked'
            else:
                y_label = st.text_input('y-axis label: ', value=y_column)
                y_labels = {kind: y_label for kind in kinds}
                settings['layout'] = 'single'
            y_label = y_labels[kinds[0]]

            row_of = {kind: i + 1 for i, kind in enumerate(kinds)}

            # Plotting
            if use_plotly:
                n_rows = len(kinds) if stacked else 1
                fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06)
                for df, label, kind in zip(data, labels, file_kinds):
                    row = row_of[kind] if stacked else 1
                    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode="lines", name=label), row=row, col=1)
                if peak_list is not None:
                    for df, label, kind in zip(peak_list, labels, file_kinds):
                        row = row_of[kind] if stacked else 1
                        fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode="markers", name=f"{label} peaks"), row=row, col=1)
                if stacked:
                    for kind in kinds:
                        fig.update_yaxes(title_text=y_labels[kind], row=row_of[kind], col=1)
                    fig.update_xaxes(title_text=x_label, row=n_rows, col=1)
                    fig.update_layout(height=350 * n_rows)
                else:
                    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                st.plotly_chart(fig)
            else:
                n_rows = len(kinds) if stacked else 1
                fig, axes = plt.subplots(n_rows, 1, sharex=True, squeeze=False, figsize=(6.4, 3.6 * n_rows))
                axes = axes[:, 0]
                ax_of = {kind: (axes[row_of[kind] - 1] if stacked else axes[0]) for kind in kinds}
                for df, label, kind in zip(data, labels, file_kinds):
                    ax_of[kind].plot(df[x_column], df[y_column], label=label)
                if peak_list is not None:
                    for df, label, kind in zip(peak_list, labels, file_kinds):
                        ax_of[kind].plot(df[x_column], df[y_column], 'o', label=f"{label} peaks")
                for kind in kinds:
                    ax_of[kind].set_ylabel(y_labels[kind])
                    ax_of[kind].legend()
                axes[-1].set_xlabel(x_label)
                fig.tight_layout()
                st.pyplot(fig)

            # Saving: one x/y column pair per file, side by side
            st.markdown("### Output options")
            wide = pd.concat(
                [df[[x_column, y_column]].reset_index(drop=True)
                   .rename(columns={x_column: f"{label} {x_column}", y_column: f"{label} {y_column}"})
                 for df, label in zip(data, labels)], axis=1)
            st.write(wide)
            filename = st.text_input("Filename:", value="data", key="separate_x_filename")
            write_excel(wide, filename)

            # Notebook export (multi-DataFrame version)
            add_echem_notebook_download_buttons(
                data=data,
                settings=settings,
                labels=labels,
                x_label=x_label,
                y_label=y_label,
                x_column=x_column,
                y_column=y_column,
                title="Raman / IR Spectroscopy Analysis",
                filename_base=filename
            )

if __name__ == "__main__":
    run()
