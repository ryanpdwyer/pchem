"""
Notebook export utilities for generating Jupyter notebooks from Streamlit apps.

Supports both .ipynb format and percent-script .py format (compatible with
VS Code, Spyder, and JupyterLab).
"""
import json
import io
import base64
import streamlit as st
import pandas as pd


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV string for embedding in notebooks."""
    return df.to_csv(index=False)


def create_ipynb(cells: list[dict]) -> str:
    """
    Create an .ipynb notebook from a list of cells.

    Each cell should be a dict with:
        - 'type': 'code' or 'markdown'
        - 'source': str or list of strings
    """
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": []
    }

    for i, cell in enumerate(cells):
        source = cell['source']
        if isinstance(source, str):
            source = source.split('\n')
        # Add newlines to all but last line
        source = [line + '\n' if j < len(source) - 1 else line
                  for j, line in enumerate(source)]

        nb_cell = {
            "id": f"cell-{i}",
            "cell_type": cell['type'],
            "metadata": {},
            "source": source,
        }
        if cell['type'] == 'code':
            nb_cell["outputs"] = []
            nb_cell["execution_count"] = None

        notebook["cells"].append(nb_cell)

    return json.dumps(notebook, indent=2)


def create_percent_script(cells: list[dict]) -> str:
    """
    Create a percent-script .py file from a list of cells.

    Uses # %% for code cells and # %% [markdown] for markdown cells.
    Compatible with VS Code, Spyder, JupyterLab, and PyCharm.
    """
    lines = []

    for cell in cells:
        source = cell['source']
        if isinstance(source, list):
            source = '\n'.join(source)

        if cell['type'] == 'markdown':
            lines.append('# %% [markdown]')
            # Convert markdown to comments
            for line in source.split('\n'):
                lines.append(f'# {line}')
        else:
            lines.append('# %%')
            lines.append(source)

        lines.append('')  # Empty line between cells

    return '\n'.join(lines)


def generate_spectra_notebook(
    combined_data: pd.DataFrame,
    settings: dict,
    labels: list[str],
    x_label: str,
    y_label: str,
    x_column: str,
    title: str = "Spectroscopy Data Analysis",
    data_type: str = "UV-Vis"
) -> list[dict]:
    """
    Generate notebook cells for spectroscopy data analysis.

    Returns a list of cells that can be passed to create_ipynb or create_percent_script.
    """
    csv_data = dataframe_to_csv_string(combined_data)

    cells = []

    # Title cell
    cells.append({
        'type': 'markdown',
        'source': f"# {title}\n\nGenerated from the {data_type} analysis tool."
    })

    # Imports cell
    cells.append({
        'type': 'code',
        'source': """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# For Google Colab, uncomment if needed:
# !pip install plotly
import plotly.express as px"""
    })

    # Data cell
    cells.append({
        'type': 'markdown',
        'source': "## Load Data\n\nThe data is embedded directly in this notebook."
    })

    # Embed data as CSV string
    cells.append({
        'type': 'code',
        'source': f'''# Embedded data from Streamlit app
csv_data = """
{csv_data}"""

df = pd.read_csv(StringIO(csv_data))
df.head()'''
    })

    # Settings cell
    settings_str = json.dumps(settings, indent=2)
    cells.append({
        'type': 'markdown',
        'source': "## Analysis Settings\n\nThese are the settings used in the Streamlit app:"
    })

    cells.append({
        'type': 'code',
        'source': f'''# Settings from Streamlit app
settings = {settings_str}
x_column = "{x_column}"
x_label = "{x_label}"
y_label = "{y_label}"
labels = {labels}'''
    })

    # Matplotlib plotting cell
    cells.append({
        'type': 'markdown',
        'source': "## Plot with Matplotlib"
    })

    cells.append({
        'type': 'code',
        'source': f'''fig, ax = plt.subplots(figsize=(10, 6))

x_data = df[x_column].values
for col, label in zip(df.columns[1:], labels):
    ax.plot(x_data, df[col].values, label=label)

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()'''
    })

    # Plotly plotting cell
    cells.append({
        'type': 'markdown',
        'source': "## Interactive Plot with Plotly"
    })

    cells.append({
        'type': 'code',
        'source': f'''fig = px.line(df, x=x_column, y=df.columns[1:].tolist(),
              labels={{'value': y_label, x_column: x_label}})
fig.update_layout(title="{title}")
fig.show()'''
    })

    # Export cell
    cells.append({
        'type': 'markdown',
        'source': "## Export Data\n\nSave the processed data to a file:"
    })

    cells.append({
        'type': 'code',
        'source': '''# Save to CSV
df.to_csv("processed_data.csv", index=False)

# Save to Excel
df.to_excel("processed_data.xlsx", index=False)

print("Data saved!")'''
    })

    # Customization cell
    cells.append({
        'type': 'markdown',
        'source': """## Customize Your Analysis

Feel free to modify the code above to:
- Change plot colors and styles
- Add annotations
- Perform additional calculations
- Apply different normalizations"""
    })

    return cells


def add_notebook_download_buttons(
    combined_data: pd.DataFrame,
    settings: dict,
    labels: list[str],
    x_label: str,
    y_label: str,
    x_column: str,
    title: str = "Spectroscopy Data Analysis",
    data_type: str = "UV-Vis",
    filename_base: str = "analysis"
):
    """
    Add download buttons for notebook export to a Streamlit app.

    Call this function where you want the export buttons to appear.
    """
    st.markdown("### Export as Notebook")
    st.markdown("Download a Jupyter notebook with your data and analysis code:")

    # Generate cells
    cells = generate_spectra_notebook(
        combined_data=combined_data,
        settings=settings,
        labels=labels,
        x_label=x_label,
        y_label=y_label,
        x_column=x_column,
        title=title,
        data_type=data_type
    )

    col1, col2 = st.columns(2)

    with col1:
        # .ipynb download
        ipynb_content = create_ipynb(cells)
        st.download_button(
            label="Download .ipynb",
            data=ipynb_content,
            file_name=f"{filename_base}.ipynb",
            mime="application/x-ipynb+json",
            help="Jupyter notebook format - open in JupyterLab, Colab, or VS Code"
        )

    with col2:
        # .py percent-script download
        py_content = create_percent_script(cells)
        st.download_button(
            label="Download .py",
            data=py_content,
            file_name=f"{filename_base}.py",
            mime="text/x-python",
            help="Percent-script format - open in VS Code, Spyder, or JupyterLab"
        )
