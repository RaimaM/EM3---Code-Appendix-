import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use the 'path' variable from kagglehub.dataset_download which correctly points to the dataset
base_path = r"C:\Users\Siri_\OneDrive\Skrivebord\EM3\OpenScience\archive"

# Load all subjects
all_subjects = {}
for subj_id in range(1,25): # 24 subjects total
    subj_file_path = os.path.join(base_path, f"{subj_id}.csv", f"{subj_id}.csv")
    if os.path.exists(subj_file_path):
        all_subjects[subj_id] = pd.read_csv(subj_file_path, header=None)
        print(f"Loaded subject {subj_id}: {all_subjects[subj_id].shape}")
    else:
        print(f"Mangler subject {subj_id}")


# Load all subjects into one dataframe
# Remove last 6 columns for each subject

dfs0 = []
for subj_id, df in all_subjects.items():
    temp = df.iloc[:, :68].copy()
    dfs0.append(temp)

all_data0 = pd.concat(dfs0, ignore_index=True)
all_data0.columns = ['subject', 'condition', 'trial', 'sample'] + [f'ch_{i}' for i in range(64)]
print(all_data0.shape)
print(all_data0.head())

all_data0.to_parquet(r"C:\Users\Siri_\OneDrive\Skrivebord\EM3\OpenScience\all_data0.parquet") # smaller & faster than CSV
# or if you prefer CSV:
# all_data0.to_csv("all_data0.csv", index=False)
