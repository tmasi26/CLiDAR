import re
import numpy as np
import pandas as pd
import os

# ---------- user settings ----------
file_name = "20240220_20240220_Key_Biscayne"
folder = r"C:\Users\tessa\Desktop\CLiDAR_2025"
file_path = os.path.join(folder, file_name + ".pfn")

# read csv
df = pd.read_csv(file_path, delimiter=",", skiprows=6)

# select only columns that look like "180.000000[440nm]" ... "0.000000[1020nm]"
wavelength_cols = [c for c in df.columns if ("[" in c and "nm" in c and re.match(r'^\s*[0-9.+-]', c))]
df_filtered = df[wavelength_cols]

# compute per-column means (preserves column order)
col_means = df_filtered.mean(axis=0)

# ---------- split into sets when a large drop (jump) occurs ----------
sets = [] #this holds ALL groups, a group in a group
current_set = [] # this holds just the current group

# detection params (tweak if needed)
ratio_threshold = 20.0    # previous_mean / current_mean > this → consider a jump
min_small_val = 5.0       # ensure current mean is relatively small (to avoid false splits)

#
prev_val = None
#col_means maps from column name to a mean value 
#for Series, it has (index, value) pairs, in this case (col_name, mean_val)
    #col_name example "180.0000000[440nm]"
    #mean_val example "822.7"
#The .items() on a Series loops over (index, value) pairs, so each iteration gives one column and its mean
for col_name, mean_val in col_means.items():
    if prev_val is not None:
        # avoid dividing by zero
        ratio = prev_val / (mean_val + 1e-12)
        if ratio > ratio_threshold and mean_val < min_small_val:
            # big jump detected → start new set
            sets.append(current_set) #saves a finished group
            current_set = [] #starts a new group
    current_set.append(col_name)
    prev_val = mean_val

# append last set
if current_set:
    sets.append(current_set)

# sanity check: expect 4 sets
if len(sets) < 4:
    print(f"Warning: detected only {len(sets)} sets (expected 4). Check thresholds.")
elif len(sets) > 4:
    print(f"Warning: detected {len(sets)} sets (expected 4). Will use first 4 sets.")


# ---------- build Series for each set using angle from header ----------
series_list = [] #list of lists, each inner list is one wavelength's set of solumns (all angles for 440nm)
col_labels = [] # one group fo column names (from the OG headers)
for s_idx, col_list in enumerate(sets): #s_idx is the group index (0, 1, 2,...)
    # extract wavelength label (e.g., '440nm') from the first column in this set
        #looks at the first column header in the group (ex. 180.000[440nm])
        #uses regex '\[(\d+)\s*nm\] to grab the wavelength (digits inside [...])
            #if the regex fails, it creates a fallback name like "set1"
            #Stores the label for later
    first_col = col_list[0]
    m_wl = re.search(r'\[(\d+)\s*nm\]', first_col)
    if m_wl:
        wl_label = f"{m_wl.group(1)}nm"
    else:
        wl_label = f"set{s_idx+1}"
    col_labels.append(wl_label)

    # build mapping angle -> mean for this set
        #re.match(r'^\s*[0-9.+-]+', col) looks for the number at the start of the header (the scattering angle)
        #converts it from string to a float
        #col_means[col] gives the mean value for that column
    angle_values = []
    mean_values = []
    for col in col_list:
        # extract the numeric angle at the start of the header, e.g. "180.000000"
        m_angle = re.match(r'^\s*([0-9.+-]+)', col)
        if not m_angle:
            # fallback: use sequential index if angle not parseable
            angle = np.nan
        else:
            angle = float(m_angle.group(1))
        #this creates two aligned lists
        angle_values.append(angle)
        mean_values.append(col_means[col])

    # flip order so it goes 0 -> 180 (ascending)
    # current order is assumed 180 -> 0, so reverse arrays
    angle_values = angle_values[::-1]
    mean_values = mean_values[::-1]

    # create Series indexed by angle
    ser = pd.Series(data=mean_values, index=angle_values, name=wl_label, dtype=float)
    # remove NaN angle entries (if any)
    ser = ser[~ser.index.isnull()]
    series_list.append(ser)

# ---------- combine into DataFrame with union of angles, sorted ascending ----------
df_combined = pd.concat(series_list, axis=1)

# sort index ascending (0 -> 180)
df_combined = df_combined.sort_index()

# add a nicely formatted angle column as first column (two decimals)
df_combined = df_combined.reset_index().rename(columns={'index': 'Scattering Angle (deg)'})
df_combined['Scattering Angle (deg)'] = df_combined['Scattering Angle (deg)'].map(lambda x: f"{x:.2f}")

# move angle col to front (already is)
# ---------- save to Excel with filename included ----------
out_name = f"{file_name}_grouped_by_wavelength_with_angles.xlsx"
out_path = os.path.join(folder, out_name)
df_combined.to_excel(out_path, index=False)

print(f"Saved grouped file to: {out_path}")
print("Columns written:", df_combined.columns.tolist())
