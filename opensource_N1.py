import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

all_data = pd.read_parquet(r"C:\Users\Siri_\OneDrive\Skrivebord\EM3\OpenScience\all_data0.parquet")

sfreq = 1024  # Hz
tmin = -1.5   # epoch start in seconds
t0_sample = 1537  # sample index of stimulus onset
n_samples = 3072  # samples per trial


# Time axis for plotting (-1.5s to +1.5s)
times = np.linspace(tmin, tmin + (n_samples - 1) / sfreq, n_samples)

ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3',
            'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1',
            'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',
            'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4',
            'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',
            'P10', 'PO8', 'PO4', 'O2', 'VEOa', 'VEOb', 'HEOL', 'HEOR', 'Nose', 'TP10']

def get_subject_erps(npdata, conditions_of_interest=(1, 2)):
    npdata = np.array(npdata)

    n_trials = len(np.unique(npdata[:, 1]))
    onset_rows = np.where(npdata[:, 3] == 1)[0]
    trial_conditions = npdata[onset_rows, 2].astype(int)

    n_trials = len(onset_rows)
    EEGdata = npdata.reshape(n_trials, n_samples, npdata.shape[1])
    EEGdata = EEGdata[:, :, 4:]  # ← was 4:68, but now there are only 68 columns total so just take from 4 onwards
    EEGdata = np.swapaxes(EEGdata, 1, 2)

    baseline_samples = slice(0, t0_sample)
    baseline_mean = EEGdata[:, :, baseline_samples].mean(axis=2, keepdims=True)
    EEGdata = EEGdata - baseline_mean

    erps = {}
    for cond in conditions_of_interest:
        mask = trial_conditions == cond
        if mask.sum() > 0:
            erps[cond] = EEGdata[mask].mean(axis=0)

    return erps

# Condition labels
condition_names = {1: 'button_tone', 2: 'playback_tone', 3: 'button_alone'}

all_erps = {}
for subj_id, df in all_data.groupby('subject'):
    print(f"Processing subject {subj_id}...")
    all_erps[subj_id] = get_subject_erps(df)


# DEFINE N1 CHANNELS AND TIME WINDOW

# Auditory N1 is strongest at these central channels
n1_channels = ['Fz', 'FCz', 'Cz', 'FC3', 'FC4']
n1_indices = [ch_names.index(ch) for ch in n1_channels]

# N1 window: 80-150ms post-stimulus
n1_tmin, n1_tmax = 0.08, 0.15
n1_mask = (times >= n1_tmin) & (times <= n1_tmax)

# EXTRACT N1 AMPLITUDE PR SUBJECT
n1_amplitudes = {}  # {subj_id: {condition: mean_N1_amplitude}}

for subj_id, erps in all_erps.items():
    n1_amplitudes[subj_id] = {}
    for cond, erp in erps.items():
        # Average over N1 channels, then over N1 time window
        n1_amplitudes[subj_id][cond] = erp[n1_indices, :][:, n1_mask].mean()


# PLOT N1

# Average across all subjects per condition
grand_avg = {}
for cond in (1, 2):
    grand_avg[cond] = np.mean(
        [all_erps[s][cond] for s in all_erps if cond in all_erps[s]], axis=0
    )

# Plot at Fz
#fz_idx = ch_names.index('Fz')

plt.figure(figsize=(10, 4))
# Plot average across N1 channels (same as stats)?
plt.plot(times, grand_avg[1][n1_indices, :].mean(axis=0), label='Button tone')
plt.plot(times, grand_avg[2][n1_indices, :].mean(axis=0), label='Playback tone')
plt.axvline(0, color='k', linestyle='--', label='Stimulus onset')
plt.axvspan(0.08, 0.15, alpha=0.2, color='red', label='N1 window')
plt.xlim(-0.2, 0.5)  # zoom in around stimulus
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title('Grand Average ERP (Fz, FCz, Cz, FC3, FC4)')
plt.legend()
plt.grid(True)
plt.show()

# STATS

# Get N1 per subject for each condition
subj_ids = [s for s in n1_amplitudes if 1 in n1_amplitudes[s] and 2 in n1_amplitudes[s]]
button_n1    = [n1_amplitudes[s][1] for s in subj_ids]
playback_n1  = [n1_amplitudes[s][2] for s in subj_ids]

t_stat, p_val = stats.ttest_rel(button_n1, playback_n1)
print(f"N1 suppression: t={t_stat:.3f}, p={p_val:.3f}")
print(f"Mean button_tone N1:   {np.mean(button_n1):.4f} µV")
print(f"Mean playback_tone N1: {np.mean(playback_n1):.4f} µV")

# Fz only:
    # N1 suppression: t=1.727, p=0.098
    # Mean button_tone N1:   -2.3099 µV
    # Mean playback_tone N1: -3.1265 µV

# Average of Fz, FCz, Cz, FC3, FC4 (channels used in the original dataset):
    # N1 suppression: t=3.164, p=0.004
    # Mean button_tone N1:   -2.0160 µV
    # Mean playback_tone N1: -3.1113 µV

# Plot for one subject
subj = 1
plt.figure(figsize=(10, 4))
plt.plot(times, all_erps[subj][1][n1_indices, :].mean(axis=0), label='Button tone')
plt.plot(times, all_erps[subj][2][n1_indices, :].mean(axis=0), label='Playback tone')
plt.axvline(0, color='k', linestyle='--', label='Stimulus onset')
plt.axvspan(0.08, 0.15, alpha=0.2, color='red', label='N1 window')
plt.xlim(-0.2, 0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (µV)')
plt.title(f'ERP at (Fz, FCz, Cz, FC3, FC4) - Subject {subj}')
plt.legend()
plt.grid(True)
plt.show()