import mne
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy import signal
from dash import Dash, dcc, html, Input, Output

# Load the EDF file
edf_file = "abnormal/01_tcp_ar/aaaaahte_s003_t000.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

raw.filter(0.5, 49, fir_design='firwin')

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

overall_min = np.inf
overall_max = -np.inf

for channel in channel_names:
    channel_data = eeg_data.loc[channel].values
    
    # Compute the spectrogram for the current channel
    _, _, Sxx = signal.spectrogram(channel_data, fs=raw.info['sfreq'], nperseg=256, noverlap=128)
    Sxx = np.squeeze(Sxx)
    
    # Update the overall minimum and maximum spectral power values
    overall_min = min(overall_min, np.log(Sxx).min())
    overall_max = max(overall_max, np.log(Sxx).max())

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
    dcc.Graph(id='spectrogram-graph'),
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
    Output('spectrogram-graph', 'figure'),
    Input('channel-dropdown', 'value'),
    Input('time-slider', 'value')
)
def update_spectrogram(selected_channel, time_range):
    start, end = time_range

    # Subset the data based on the selected channel and time range
    subset_data = eeg_data.loc[selected_channel, times[start:end]]

    # Compute the spectrogram
    f, t, Sxx = signal.spectrogram(subset_data.values, fs=raw.info['sfreq'], nperseg=256, noverlap=128)
    Sxx = np.squeeze(Sxx)

    # Create a Plotly spectrogram trace
    spectrogram_trace = go.Heatmap(
        z=np.log(Sxx),
        x=t,
        y=f,
        zmax=overall_max,
        zmin=overall_min,
        colorscale='Inferno',
        colorbar=dict(title='Spectral Power (dB)'),
    )

    # Create the Plotly figure
    fig = go.Figure(data=spectrogram_trace)

    # Update the layout
    fig.update_layout(
        title=f'EEG Spectrogram ({selected_channel})',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        xaxis=dict(
            tickvals=subset_data.index[::len(subset_data) // 10],
            ticktext=[f'{t:.3f}' for t in subset_data.index[::len(subset_data) // 10]],  # Tick labels
        ),
        yaxis=dict(range=[0, 50])
    )

    return f'Selected Time Range: {times[start]:.3f} - {times[end]:.3f}', fig

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)