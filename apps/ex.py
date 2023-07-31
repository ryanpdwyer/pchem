# Plot a trisurf with the same color everywhere using plotly
# 
import plotly.figure_factory as ff 

import numpy as np  
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

fig = ff.create_trisurf(x=df['x'], y=df['y'], z=df['z'], colormap=['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,176,246)'], simplices=df['simplices'], title="Mt Bruno Elevation", aspectratio=dict(x=1, y=1, z=0.5), showbackground=True, backgroundcolor="rgb(230, 230,230)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)")

# Change the z axis limits to -2, 30
fig.update_layout(scene_zaxis_range=[-2, 30])

# Add another trisurf to the same figure with a different color



