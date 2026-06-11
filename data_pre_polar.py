"""
The following preprocessing steps are done: 
1. dataframes are pivotted to have the columns timestamp, building, meter value
2. Weatherdata is joined on timestamp and building site
3. Metadata is loaded, scaled globally, categorical columns are ordinal encoded and the data is joined to the main set
4. Values missing in between data (e.g. by meter malfunction) are interpolated
5. Day of week, week of month, month of year and days since start are added
5. Numerical values (inc. day features) are standard scaled per building
"""

import polars as pl
import os

# Function to pivot tables
def process_wide_meter_file(file_path, meter_name):
    return (
        pl.scan_csv(file_path)
        .unpivot(index="timestamp", variable_name="building", value_name=meter_name)
        .with_columns([
            pl.col("timestamp").cast(pl.String),
            pl.col("building").cast(pl.String),
            pl.col(meter_name).cast(pl.Float32)
        ])
    )

df = process_wide_meter_file('electricity_cleaned.csv', 'electricity')

meter_files = {
    'water_cleaned.csv': 'water', 'steam_cleaned.csv': 'steam',
    'irrigation_cleaned.csv': 'irrigation', 'gas_cleaned.csv': 'gas',
    'hotwater_cleaned.csv': 'hotwater', 'chilledwater_cleaned.csv': 'chilledwater',
    'solar_cleaned.csv': 'solar'
}


for file_name, meter_name in meter_files.items():
    temp_df = process_wide_meter_file(file_name, meter_name)
    df = df.join(temp_df, on=['timestamp', 'building'], how='full', coalesce=True)

weather_df = pl.scan_csv('weather.csv').with_columns([
    pl.col('timestamp').cast(pl.String), pl.col('site_id').cast(pl.String)
])
df = df.with_columns(pl.col('building').str.split('_').list.get(0).alias('site_id'))
df = df.join(weather_df, on=['timestamp', 'site_id'], how='left', coalesce=True)


meta_df = pl.read_csv('metadata.csv').rename({'building_id': 'building'})

# 1. Drop unwanted columns
cols_to_drop = ['date opened', 'lat', 'lng', 'building_id_kaggle', 'site_id_kaggle', 'sqft']
meta_df = meta_df.drop([c for c in cols_to_drop if c in meta_df.columns])

# Find numeric vs string columns excluding building as that is the join key
numeric_cols = [
    col for col, dtype in zip(meta_df.columns, meta_df.dtypes)
    if dtype in (pl.Float32, pl.Float64, pl.Int64, pl.Int32) and col != 'building'
]

string_cols = [
    col for col, dtype in zip(meta_df.columns, meta_df.dtypes)
    if dtype in (pl.String, pl.Utf8) and col != 'building'
]

# Global Standard Scaling for ALL numeric features (sqm, floors, yearbuilt, etc.)
scale_exprs = []
for col in numeric_cols:
    expr = ((pl.col(col).cast(pl.Float32) - pl.col(col).mean()) / pl.col(col).std()).alias(col)
    scale_exprs.append(expr)

if scale_exprs:
    meta_df = meta_df.with_columns(scale_exprs)

# Ordinal Encoding
with pl.StringCache():
    encoded_exprs = []
    for col in string_cols:
        # +1 to reserve 0 for nulls
        expr = (pl.col(col).cast(pl.Categorical).to_physical() + 1).fill_null(0).cast(pl.Int32).alias(col)
        encoded_exprs.append(expr)

    if encoded_exprs:
        meta_df = meta_df.with_columns(encoded_exprs)

# Join metadata
df = df.join(meta_df.lazy(), on='building', how='left', coalesce=True)

print("Phase 1: Streaming raw joined data to disk...")
df.sink_parquet('staged_raw_data.parquet')
print("Phase 1 Complete.")

# =============== Split into two parts to excecute in a notebook if necessary for RAM ==================

split_date = pl.datetime(2017, 6, 1)
columns_to_interpolate = [
    'electricity', 'water', 'steam', 'gas', 'hotwater', 'chilledwater', 'solar',
    'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR',
    'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed'
]
features_to_scale = ['water', 'steam', 'gas', 'hotwater', 'chilledwater', 'solar',
    'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 
    'windDirection', 'windSpeed', 'hour', 'day_of_week', 'month', 'days_from_start']

print("Extracting unique buildings...")
unique_buildings = pl.read_parquet('staged_raw_data.parquet', columns=['building'])['building'].unique().to_list()

dataset_dir = 'final_lstm_dataset'
os.makedirs(dataset_dir, exist_ok=True)

print(f"Phase 2: Processing {len(unique_buildings)} buildings chunk-by-chunk...")

# Processing per building to save RAM
for i, bldg in enumerate(unique_buildings):
    bldg_df = pl.scan_parquet('staged_raw_data.parquet').filter(pl.col('building') == bldg)

    bldg_df = bldg_df.with_columns(pl.col('timestamp').str.to_datetime("%Y-%m-%d %H:%M:%S"))
    bldg_df = bldg_df.with_columns([
        pl.col('timestamp').dt.hour().cast(pl.Int8).alias('hour'),
        pl.col('timestamp').dt.weekday().cast(pl.Int8).alias('day_of_week'),
        pl.col('timestamp').dt.month().cast(pl.Int8).alias('month'),
        (pl.col('timestamp') - pl.col('timestamp').min()).dt.total_days().cast(pl.Int16).alias('days_from_start')
    ])

    # Skip building if there are missing values in the target variable
    if bldg_df.filter(pl.col('electricity').is_null()).shape[0] > 0:
        continue
    

    bldg_df = bldg_df.sort('timestamp').with_columns([
        pl.col(col).interpolate().forward_fill() for col in columns_to_interpolate
    ])

    # Scale features per building
    scale_exprs = []
    for col in features_to_scale:
        train_mean = pl.when(pl.col('timestamp') < split_date).then(pl.col(col)).otherwise(None).mean()
        train_std = pl.when(pl.col('timestamp') < split_date).then(pl.col(col)).otherwise(None).std()
        safe_std = pl.when(train_std == 0).then(1e-6).otherwise(train_std)
        scale_exprs.append(((pl.col(col) - train_mean) / safe_std).alias(col))

    bldg_df = bldg_df.with_columns(scale_exprs)

    processed_chunk = bldg_df.collect()
    
    safe_bldg_name = bldg.replace("/", "_") # Just in case there are slashes in the name
    file_path = os.path.join(dataset_dir, f"{safe_bldg_name}.parquet")
    processed_chunk.write_parquet(file_path)

    if (i + 1) % 50 == 0:
        print(f"Processed {i + 1} / {len(unique_buildings)} buildings...")

print("Clean partitioned dataset created.")