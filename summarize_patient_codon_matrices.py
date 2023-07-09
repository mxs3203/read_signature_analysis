import glob
import tqdm
import pandas as pd

files = glob.glob("data/*.csv")

all_samples_freq = pd.DataFrame()
for f in tqdm.tqdm(files):
    freq_df = pd.read_csv(f)
    id = f.split('/')[-1].split('.')[0]  # current ID
    id = id.split("_")[0] +"_"+ id.split("_")[1]
    medians = freq_df.median(axis=0)
    medians['id'] = id
    all_samples_freq = pd.concat([all_samples_freq, medians.to_frame().T], ignore_index=True)

all_samples_freq.to_csv("median_frequencies_by_id.csv")