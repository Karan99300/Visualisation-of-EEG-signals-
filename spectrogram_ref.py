import mne
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy import signal
from dash import Dash, dcc, html, Input, Output, State, callback_context
from plotly.subplots import make_subplots

# Load the EDF file
edf_file = "abnormal/01_tcp_ar/aaaaahte_s003_t000.edf"
raw = mne.io.read_raw_edf(edf_file, preload=True)

raw.filter(0.5, 30, fir_design='firwin')

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

# Calculate overall min and max spectral power
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
        value=[channel_names[0]],
        multi=True
    ),
    dcc.Graph(id='spectrogram-graph'),
    html.Div([
        html.Label('Select time range:'),
        html.Div([
            dcc.Input(id='start-time-input', type='number', value=times[0], min=times[0], max=times[-1], step=times[1]-times[0], style={'width': '100px', 'margin-right': '20px'}),
            dcc.Input(id='end-time-input', type='number', value=times[len(times) // 10], min=times[0], max=times[-1], step=times[1]-times[0], style={'width': '100px', 'margin-right': '20px'}),
        ], style={'margin-bottom': '20px'}),
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
    Output('start-time-input', 'value'),
    Output('end-time-input', 'value'),
    Output('time-slider', 'value'),
    Output('slider-output-container', 'children'),
    Output('spectrogram-graph', 'figure'),
    Input('time-slider', 'value'),
    Input('start-time-input', 'value'),
    Input('end-time-input', 'value'),
    Input('channel-dropdown', 'value')
)
def sync_and_update(time_range, start_time, end_time, selected_channels):
    ctx = callback_context

    start_idx = time_range[0]
    end_idx = time_range[1]

    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id']
        
        if 'time-slider' in prop_id:
            start_time = times[start_idx]
            end_time = times[end_idx]
        else:
            start_idx = np.searchsorted(times, start_time)
            end_idx = np.searchsorted(times, end_time)

    spectrograms = []

    for selected_channel in selected_channels:
        # Subset the data based on the selected channel and time range
        subset_data = eeg_data.loc[selected_channel, times[start_idx:end_idx]]

        # Compute the spectrogram
        f, t, Sxx = signal.spectrogram(subset_data.values, fs=raw.info['sfreq'], nperseg=256, noverlap=128)
        Sxx = np.squeeze(Sxx)
        Sxx_log = np.log(Sxx)
        
        # Collect the spectrogram for plotting later
        spectrograms.append((selected_channel, f, t, Sxx_log))

    # Create subplots
    fig = make_subplots(
        rows=len(selected_channels), cols=1,
        subplot_titles=[f'Spectrogram - {channel}' for channel in selected_channels]
    )

    for i, (selected_channel, f, t, Sxx_log) in enumerate(spectrograms):
        # Create a Plotly spectrogram trace
        spectrogram_trace = go.Heatmap(
            z=Sxx_log,
            x=t + times[start_idx],  # Adjust x-axis range
            y=f,
            zmax=overall_max,
            zmin=overall_min,
            colorscale='Inferno',
            colorbar=dict(title=f'Spectral Power (dB)'),
        )

        # Add the trace to the subplot
        fig.add_trace(spectrogram_trace, row=i + 1, col=1)

        # Update the layout for each subplot
        fig.update_xaxes(title_text='Time (s)', row=i + 1, col=1)
        fig.update_yaxes(title_text='Frequency (Hz)', range=[0.5, 30], row=i + 1, col=1)

    fig.update_layout(
        title=f'EEG Spectrograms',
        height=500 * len(selected_channels),  # Adjust height based on the number of subplots
    )

    return start_time, end_time, [start_idx, end_idx], f'Selected Time Range: {start_time:.3f} - {end_time:.3f}', fig

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=True)
