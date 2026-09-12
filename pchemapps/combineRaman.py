
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


PEAK_DIRECTIONS = ("Up (Raman, absorbance)", "Down (% transmittance)")


def find_candidate_peaks(df, x_column, y_column, direction, min_prominence_frac=0.05, max_candidates=15):
    """Candidate peaks by prominence (scipy.signal.find_peaks), most prominent first.

    direction: "Up..." finds maxima, "Down..." finds minima (e.g. % transmittance).
    min_prominence_frac: minimum prominence as a fraction of the y range of the file."""
    x = df[x_column].values.astype(float)
    y = df[y_column].values.astype(float)
    yy = -y if direction.startswith("Down") else y
    yrange = np.nanmax(yy) - np.nanmin(yy)
    if not np.isfinite(yrange) or yrange == 0:
        return pd.DataFrame({x_column: [], y_column: [], "Prominence": []})
    idx, props = signal.find_peaks(yy, prominence=min_prominence_frac * yrange)
    order = np.argsort(props["prominences"])[::-1][:max_candidates]
    idx = idx[order]
    out = pd.DataFrame({x_column: x[idx], y_column: y[idx],
                        "Prominence": props["prominences"][order] / yrange})
    return out.reset_index(drop=True)


PALETTE = px.colors.qualitative.Plotly


def trace_color(i):
    return PALETTE[i % len(PALETTE)]


def style_axes_plotly(fig, grid=True):
    """Vertical major + minor gridlines (optional) and a hover spike, so peaks can be compared across traces / panels."""
    fig.update_xaxes(showgrid=grid, gridcolor="rgba(0,0,0,0.18)", ticks="outside",
                     minor=dict(showgrid=grid, gridcolor="rgba(0,0,0,0.06)"),
                     showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikedash="dot", spikecolor="gray")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig.update_layout(plot_bgcolor="white", hovermode="x",
                      legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="left", x=0),
                      margin=dict(l=60, r=20, t=90, b=50))
    return fig


def add_peak_lines_plotly(fig, peaks_x, peak_labels, color, row=1, n_rows=1, fontsize=10):
    """Dashed vertical line at each peak spanning every panel, label written on the trace's own panel."""
    ax_id = "" if row == 1 else str(row)
    shapes = list(fig.layout.shapes or [])
    annots = list(fig.layout.annotations or [])
    for x, text in zip(peaks_x, peak_labels):
        shapes.append(dict(type="line", xref=f"x{ax_id}", yref="paper", x0=float(x), x1=float(x), y0=0, y1=1,
                           line=dict(color=color, width=1, dash="dash"), opacity=0.7))
        # top panel: label sits above the axes; lower panels: inside, hanging from the top edge
        annots.append(dict(x=float(x), y=1.0, xref=f"x{ax_id}", yref=f"y{ax_id} domain",
                           text=str(text), showarrow=False, textangle=-90, xanchor="center",
                           yanchor="bottom" if row == 1 else "top", font=dict(size=fontsize, color=color)))
    fig.update_layout(shapes=shapes, annotations=annots)
    return fig


def edit_peaks(df, x_column, y_column, label, key, n_show=3):
    """Candidate-peak table: tick **Show** to annotate a peak, edit its **Label**, add or delete rows.
    Returns only the rows that are ticked, sorted by x."""
    table = pd.DataFrame({"Show": [i < n_show for i in range(len(df))],
                          x_column: df[x_column].values.astype(float),
                          y_column: df[y_column].values.astype(float)})
    table["Label"] = [f"{x:.4g}" for x in table[x_column]]
    if "Prominence" in df:
        table["Prominence"] = df["Prominence"].values
    st.caption(label)
    cfg = {"Show": st.column_config.CheckboxColumn(default=True),
           x_column: st.column_config.NumberColumn(format="%.2f"),
           y_column: st.column_config.NumberColumn(format="%.4g", required=False),
           "Label": st.column_config.TextColumn(),
           "Prominence": st.column_config.NumberColumn(format="%.2f", disabled=True)}
    edited = st.data_editor(table, num_rows="dynamic", key=key, hide_index=True, column_config=cfg)
    edited = edited.dropna(subset=[x_column]).copy()
    edited["Show"] = edited["Show"].fillna(True).astype(bool)
    edited["Label"] = [lab if isinstance(lab, str) and lab.strip() else f"{x:.4g}"
                       for lab, x in zip(edited["Label"], edited[x_column])]
    return edited[edited["Show"]].sort_values(x_column).reset_index(drop=True)


def show_plotly(fig):
    """st.plotly_chart at full container width, across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def style_axes_mpl(ax, grid=True):
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(grid, axis="x", which="major", alpha=0.4)
    ax.grid(grid, axis="x", which="minor", alpha=0.12)
    ax.grid(axis="y", alpha=0.15)
    return ax


def add_peak_lines_mpl(axes, own_ax, peaks_x, peak_labels, color, fontsize=9):
    """Dashed line at each peak on every axis (no legend entry); label written just above the trace's own axis."""
    for x, text in zip(peaks_x, peak_labels):
        for ax in axes:
            ax.axvline(x, color=color, ls="--", lw=0.8, alpha=0.6)
        own_ax.annotate(str(text), (x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 3), textcoords="offset points", rotation=90,
                        fontsize=fontsize, color=color, ha="center", va="bottom")


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
            annot_fontsize = 9

            if st.checkbox("Pick peaks?"):
                st.markdown("""### Peak picking
Candidate peaks are found by *prominence* (how far a peak rises above the surrounding signal).
Tick **Show** for the peaks you want annotated, edit the **Label** text, delete rows you don't want,
or add a row (enter the x value) to annotate something the picker missed.""")
                # Peak direction per file type: % transmittance peaks point down
                direction_of = {}
                for kind in kinds:
                    default = 1 if "transmit" in choice_names[kind][1].lower() else 0
                    direction_of[kind] = st.selectbox(f"Peak direction ({kind})", PEAK_DIRECTIONS, index=default,
                                                      key=f"peakdir_{kind}")
                c1, c2, c3 = st.columns(3)
                min_prom = c1.slider("Min. prominence (% of y range)", 1, 50, 5) / 100
                max_cand = c2.number_input("Candidates per file", 1, 50, 15)
                n_show = c3.number_input("Shown by default", 0, 50, 3)
                annot_fontsize = st.number_input("Peak label font size", 4, 24, 9)
                settings['peak_label_fontsize'] = annot_fontsize
                settings['peak_min_prominence'] = min_prom
                settings['peak_directions'] = [direction_of[d.kind] for d in loaded]

                peak_list = []
                for i, (df, label, d) in enumerate(zip(data, labels, loaded)):
                    cands = find_candidate_peaks(df, x_column, y_column, direction_of[d.kind], min_prom, max_cand)
                    key = f"peaks_{i}_{direction_of[d.kind]}_{min_prom}_{max_cand}_{n_show}"
                    peak_list.append(edit_peaks(cands, x_column, y_column, label, key=key, n_show=n_show))

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
            show_grid = st.checkbox("Vertical gridlines?", value=True, key="grid_same_x")
            settings['gridlines'] = show_grid

            # Plotting
            if use_plotly:
                plotly_fig = px.line(combined_data, x=x_column, y=combined_data.columns[1:],
                        labels={'value': y_label, x_column: x_label},
                        color_discrete_sequence=PALETTE)
                style_axes_plotly(plotly_fig, show_grid)

                if peak_list is not None:
                    for i, df in enumerate(peak_list):
                        add_peak_lines_plotly(plotly_fig, df[x_column].values, df['Label'].values, trace_color(i), fontsize=annot_fontsize)

                show_plotly(plotly_fig)
            else:
                fig, ax = plt.subplots()
                for i, (col, fname, label) in enumerate(zip(combined_data.values[:, 1:].T, filenames, labels)):
                    ax.plot(x_data, col, label=label, color=f"C{i}")
                style_axes_mpl(ax, show_grid)

                if peak_list is not None:
                    for i, (df, label) in enumerate(zip(peak_list, labels)):
                        add_peak_lines_mpl([ax], ax, df[x_column].values, df['Label'].values, f"C{i}", fontsize=annot_fontsize)

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
            show_grid = st.checkbox("Vertical gridlines?", value=True, key="grid_separate_x")
            settings['gridlines'] = show_grid
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
                for i, (df, label, kind) in enumerate(zip(data, labels, file_kinds)):
                    row = row_of[kind] if stacked else 1
                    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode="lines", name=label,
                                             line=dict(color=trace_color(i))), row=row, col=1)
                style_axes_plotly(fig, show_grid)
                if peak_list is not None:
                    for i, (df, label, kind) in enumerate(zip(peak_list, labels, file_kinds)):
                        row = row_of[kind] if stacked else 1
                        add_peak_lines_plotly(fig, df[x_column].values, df['Label'].values, trace_color(i), row=row, n_rows=n_rows, fontsize=annot_fontsize)
                if stacked:
                    for kind in kinds:
                        fig.update_yaxes(title_text=y_labels[kind], row=row_of[kind], col=1)
                    fig.update_xaxes(title_text=x_label, row=n_rows, col=1)
                    fig.update_layout(height=350 * n_rows)
                else:
                    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
                show_plotly(fig)
            else:
                n_rows = len(kinds) if stacked else 1
                fig, axes = plt.subplots(n_rows, 1, sharex=True, squeeze=False, figsize=(6.4, 3.6 * n_rows))
                axes = axes[:, 0]
                ax_of = {kind: (axes[row_of[kind] - 1] if stacked else axes[0]) for kind in kinds}
                for i, (df, label, kind) in enumerate(zip(data, labels, file_kinds)):
                    ax_of[kind].plot(df[x_column], df[y_column], label=label, color=f"C{i}")
                for ax in axes:
                    style_axes_mpl(ax, show_grid)
                if peak_list is not None:
                    for i, (df, label, kind) in enumerate(zip(peak_list, labels, file_kinds)):
                        add_peak_lines_mpl(axes, ax_of[kind], df[x_column].values, df['Label'].values, f"C{i}", fontsize=annot_fontsize)
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
                filename_base=filename,
                peaks=peak_list
            )

if __name__ == "__main__":
    run()
