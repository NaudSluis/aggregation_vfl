"""
The following preprocessing steps are done: 
1. Dataframes are pivotted to have the columns timestamp, building, meter value
2. Weatherdata is joined on timestamp and building site
3. Metadata is loaded, scaled globally, categorical columns are ordinal encoded and the data is joined to the main set
4. Values missing in between data (e.g. by meter malfunction) are interpolated
5. Day of week, weekday, month of year are CYCLICALLY ENCODED (sin/cos)
6. Days since start are added
7. Continuous numerical values (inc. days_from_start) are standard scaled per building
8. All buildings are appended into one big merged table and saved to the project root
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Change to data directory (assuming you run this from within the data folder)
# If you run this from the project root, you may need to adjust this!
if os.path.exists('data'):
    os.chdir('data')

# =============== Phase 1: Load and merge all meter data ==================

def process_wide_meter_file(file_path, meter_name):
    """Load wide meter file and pivot to long format"""
    df = pd.read_csv(file_path)
    df = df.melt(id_vars=['timestamp'], var_name='building', value_name=meter_name)
    df['timestamp'] = df['timestamp'].astype(str)
    df['building'] = df['building'].astype(str)
    df[meter_name] = df[meter_name].astype('float32')
    return df

print("Loading meter data...")
df = process_wide_meter_file('electricity_cleaned.csv', 'electricity')

meter_files = {
    'water_cleaned.csv': 'water', 
    'steam_cleaned.csv': 'steam',
    'irrigation_cleaned.csv': 'irrigation', 
    'gas_cleaned.csv': 'gas',
    'hotwater_cleaned.csv': 'hotwater', 
    'chilledwater_cleaned.csv': 'chilledwater',
    'solar_cleaned.csv': 'solar'
}

for file_name, meter_name in meter_files.items():
    if os.path.exists(file_name):
        temp_df = process_wide_meter_file(file_name, meter_name)
        df = pd.merge(df, temp_df, on=['timestamp', 'building'], how='outer')

print(f"Merged meter data shape: {df.shape}")

# =============== Join Weather Data ==================

print("Loading weather data...")
weather_df = pd.read_csv('weather.csv')
weather_df['timestamp'] = weather_df['timestamp'].astype(str)
weather_df['site_id'] = weather_df['site_id'].astype(str)

# Extract site_id from building
df['site_id'] = df['building'].str.split('_').str[0]

# Prevent overlapping column names from weather_df
overlap_weather = {c: f"{c}_weather" for c in weather_df.columns if c in df.columns and c not in ['timestamp', 'site_id']}
weather_df = weather_df.rename(columns=overlap_weather)

df = pd.merge(df, weather_df, on=['timestamp', 'site_id'], how='left')
print(f"After weather join: {df.shape}")

# =============== Load and Process Metadata ==================

print("Loading metadata...")
meta_df = pd.read_csv('metadata.csv').rename(columns={'building_id': 'building'})

cols_to_drop = ['date opened', 'lat', 'lng', 'building_id_kaggle', 'site_id_kaggle', 'sqft']
meta_df = meta_df.drop(columns=[c for c in cols_to_drop if c in meta_df.columns])

numeric_cols = meta_df.select_dtypes(include=[np.number]).columns.tolist()
if 'building' in numeric_cols:
    numeric_cols.remove('building')
string_cols = meta_df.select_dtypes(include=['object', 'string']).columns.tolist()
if 'building' in string_cols:
    string_cols.remove('building')

# Global standard scaling for numeric features
for col in numeric_cols:
    mean = meta_df[col].mean()
    std = meta_df[col].std()
    meta_df[col] = (meta_df[col] - mean) / std

# Ordinal encoding for string columns
for col in string_cols:
    le = LabelEncoder()
    meta_df[col] = le.fit_transform(meta_df[col].astype(str)) + 1

# Prevent overlapping column names from meta_df
overlap_meta = {c: f"{c}_meta" for c in meta_df.columns if c in df.columns and c != 'building'}
meta_df = meta_df.rename(columns=overlap_meta)

df = pd.merge(df, meta_df, on='building', how='left')
print(f"After metadata join: {df.shape}")

# =============== Phase 2: Add Temporal Features and Process ==================

print("Processing all data...")
df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d %H:%M:%S")
df = df.sort_values(['building', 'timestamp']).reset_index(drop=True)

# ------------------------------------------------------------------
# CYCLICAL ENCODING
# ------------------------------------------------------------------
# Hour: 0-23 (Divisor: 24.0)
df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24.0).astype('float32')
df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24.0).astype('float32')

# Day of Week: 0-6 (Divisor: 7.0)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7.0).astype('float32')
df['day_of_week_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7.0).astype('float32')

# Month: 1-12 (Divisor: 12.0)
df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12.0).astype('float32')
df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12.0).astype('float32')

# Linear time progression per building
df['days_from_start'] = df.groupby('building')['timestamp'].apply(
    lambda x: (x - x.min()).dt.days.astype('float32')
).reset_index(level=0, drop=True)

print(f"Added cyclical temporal features, shape: {df.shape}")

# =============== Phase 3: Interpolation and Feature Scaling ==================

columns_to_interpolate = [
    'electricity', 'water', 'steam', 'gas', 'hotwater', 'chilledwater', 'solar',
    'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR',
    'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed'
]
features_to_scale = [
    'electricity', 'water', 'steam', 'gas', 'hotwater', 'chilledwater', 'solar',
    'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR', 
    'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed', 
    'days_from_start'
]

split_date = pd.to_datetime('2017-06-01')

unique_buildings = df['building'].unique()
print(f"Processing {len(unique_buildings)} buildings...")

processed_dfs = []

for i, bldg in enumerate(unique_buildings):
    bldg_df = df[df['building'] == bldg].copy()
    
    if bldg_df['electricity'].isna().any():
        continue
    
    # Interpolate
    for col in columns_to_interpolate:
        if col in bldg_df.columns:
            bldg_df[col] = bldg_df[col].interpolate(method='linear', limit_direction='both')
            bldg_df[col] = bldg_df[col].ffill().bfill()
    
    # Standard Scale
    train_mask = bldg_df['timestamp'] < split_date

    if 'electricity' in bldg_df.columns:
        e_mean = bldg_df.loc[train_mask, 'electricity'].mean()
        e_std = bldg_df.loc[train_mask, 'electricity'].std()
        if pd.isna(e_std) or e_std == 0:
            e_std = 1e-6
        bldg_df['target_mean'] = e_mean
        bldg_df['target_std'] = e_std
    
    for col in features_to_scale:
        if col in bldg_df.columns:
            train_mean = bldg_df.loc[train_mask, col].mean()
            train_std = bldg_df.loc[train_mask, col].std()
            
            if pd.isna(train_std) or train_std == 0:
                train_std = 1e-6
            
            bldg_df[col] = (bldg_df[col] - train_mean) / train_std
            bldg_df[col] = bldg_df[col].astype('float32')
    
    processed_dfs.append(bldg_df)
    
    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} / {len(unique_buildings)} buildings...")

# =============== Combine and Save Safely ==================

if processed_dfs:
    final_df = pd.concat(processed_dfs, ignore_index=True)
    print(f"Final merged dataset shape: {final_df.shape}")
    
    # EXPLCIT PATH: Save directly to the Snellius project root 
    # This prevents the job script from loading an outdated file.
    project_dir = os.path.expanduser('../')
    os.makedirs(project_dir, exist_ok=True)
    
    output_path = os.path.join(project_dir, 'final_lstm_dataset_merged.parquet')
    final_df.to_parquet(output_path, index=False)
    
    print(f"Data successfully standard-scaled and saved to: {output_path}")
else:
    print("No valid buildings to process!")