import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

channels = [
    'EEG F7-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG FP2-REF',
    'EEG F4-REF', 'EEG O1-REF', 'EEG T3-REF', 'EEG O2-REF',
    'EEG T6-REF', 'EEG F3-REF', 'EEG P4-REF', 'EEG FP1-REF', 'EEG T4-REF',
    'EEG T5-REF', 'EEG P3-REF', 'EEG F8-REF', 'EEG PZ-REF', 'EEG CZ-REF', 'EEG FZ-REF'
]

channel_map = {label: idx for idx, label in enumerate(channels)}

bipolar_combinations = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('FZ', 'CZ'), ('CZ', 'PZ'),
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2')
]

def create_bipolar_channels(raw, channel_map, bipolar_combinations):
    bipolar_data = {}
    channel_names = []
    for ch1, ch2 in bipolar_combinations:
        ch1_full = f"EEG {ch1}-REF"
        ch2_full = f"EEG {ch2}-REF"
        bipolar_channel_name = f"{ch1}-{ch2}"
        channel_names.append(bipolar_channel_name)
        bipolar_data[bipolar_channel_name] = raw.get_data(picks=[channel_map[ch1_full]])[0] - raw.get_data(picks=[channel_map[ch2_full]])[0]
    return pd.DataFrame(bipolar_data, index=np.arange(bipolar_data[bipolar_channel_name].shape[0]) / raw.info['sfreq']).T, channel_names

def process_edf_file(raw, channel_map, bipolar_combinations, fs):
    if raw.info['sfreq'] != fs:
        raw = raw.resample(fs)
    bipolar_data, channel_names = create_bipolar_channels(raw, channel_map, bipolar_combinations)
    return bipolar_data, channel_names

if __name__ == "__main__":
    fs = 250  # Define the sampling frequency

    # Load the EDF file
    edf_file = "abnormal/01_tcp_ar/aaaaahte_s003_t000.edf"
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    bipolar_data, channel_names = process_edf_file(raw, channel_map, bipolar_combinations, fs)
    
    # Print the channel names and bipolar data
    print(channel_names)
    print(bipolar_data)