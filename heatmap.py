import mne
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output

# Load the EDF file
edf_file = "abnormal/01_tcp_ar/aaaaahte_s003_t000.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

# Select the channels from the 10-20 system
channels_10_20_system = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
]
raw.pick_channels(channels_10_20_system)

# Extract data and times
data, times = raw[:]
channel_names = raw.ch_names
eeg_data = pd.DataFrame(data, index=channel_names, columns=times)

overall_min=eeg_data.values.min()
overall_max=eeg_data.values.max()

colorbar_min = -350
colorbar_max = 1000

# Initialize the Dash app
app = Dash(__name__)

# Calculate marks for RangeSlider
num_marks = 12  # Number of marks
marks_step = len(times) // num_marks
marks = {i * marks_step: f'{times[i * marks_step]:.2f}' for i in range(num_marks)}

app.layout = html.Div([
    dcc.Dropdown(
        id='channel-dropdown',
        options=[{'label': channel, 'value': channel} for channel in channel_names],
        value=channel_names[0]
    ),
    dcc.Graph(id='heatmap-graph'),
    html.Div([
        html.Label('Select time range:'),
        dcc.RangeSlider(
            id='time-slider',
            min=0,
            max=len(times) - 1,
            value=[0, len(times) // 10],
            marks=marks,
            allowCross=False,
            tooltip={'placement': 'top'}
        ),
        html.Div(id='slider-output-container', style={'margin-top': 20, 'font-size': '16px'})
    ], style={'margin': '20px'})
])

@app.callback(
    Output('slider-output-container', 'children'),
    Output('heatmap-graph', 'figure'),
    Input('channel-dropdown', 'value'),
    Input('time-slider', 'value')
)
def update_heatmap(selected_channel, time_range):
    start, end = time_range

    # Subset the data based on the selected channel and time range
    subset_data = eeg_data.loc[selected_channel, times[start:end]]
    
    
    """
    normalized_data = (subset_data - overall_min) / (overall_max - overall_min)
    scaled_data = normalized_data * (colorbar_max - colorbar_min) + colorbar_min
    
    fig = go.Figure(data=go.Heatmap(
        z=[scaled_data.values],
        x=scaled_data.index,
        y=[selected_channel],
        colorscale='Inferno',
        zmin=colorbar_min,  # Set the fixed minimum value for the colorbar
        zmax=colorbar_max,  # Set the fixed maximum value for the colorbar
        colorbar=dict(title="Voltage")  # Add color bar with title
    ))

    """
    
    fig = go.Figure(data=go.Heatmap(
        z=[subset_data.values],
        x=subset_data.index,
        y=[selected_channel],
        colorscale='Inferno',
        zmin=overall_min,  # Set the fixed minimum value for the colorbar
        zmax=overall_max,  # Set the fixed maximum value for the colorbar
        colorbar=dict(title="Voltage")  # Add color bar with title
    ))
    
    fig.update_layout(
        title=f'EEG Signal Amplitude Heatmap - {selected_channel}',
        xaxis_title='Time (s)',
        yaxis_title='Channel',
        xaxis=dict(
            tickvals=subset_data.index[::len(subset_data) // 10],
            ticktext=[f'{t:.3f}' for t in subset_data.index[::len(subset_data) // 10]],  # Tick labels
        )
    )

    return f'Selected Time Range: {times[start]:.3f} - {times[end]:.3f}', fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    
"""
Best solution so far: before every data read find overall minimum and maximum and this represents the range of the voltage bar.
Alternative: Normalise and scale the data within the color bar range.
"""