# %% [markdown]
# # Step 1: Data Cleaner & Merger (Optimized)
# This script prepares your Real Data for the AI Model.
# Optimizations:
# - Pre-indexes images for O(1) lookup speed.
# - Uses vectorized Pandas operations for faster cleaning.
# - detailed debug info for rejected files.

# %%
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# --- CONFIGURATION ---
SEARCH_DIRS = [".", "data"]
OUTPUT_DIR = "data/processed_real_data"
IMAGES_DIR = "data/satellite_images" 

# TARGET REGION
TARGET_REGION = "Multan" 

# Crop Season Settings (Rabi Season)
SEASON_START_MONTH = 11 # Nov
SEASON_END_MONTH = 4    # Apr

# %%
def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

# %%
def find_all_csvs():
    csv_files = []
    print(f"Scanning for CSV files in: {SEARCH_DIRS} ...")
    
    for search_dir in SEARCH_DIRS:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(".csv"):
                    full_path = os.path.join(root, file)
                    csv_files.append(full_path)
    
    csv_files = list(set(csv_files))
    return csv_files

# %%
def process_yield_data(all_csvs):
    print("\n[1/3] Processing Yield Data...")
    
    yield_file = None
    for f in all_csvs:
        if "yield_data.csv" in os.path.basename(f):
            yield_file = f
            break
            
    if not yield_file:
        print("ERROR: yield_data.csv not found!")
        return None

    print(f" -> Using Yield File: {yield_file}")
    df = pd.read_csv(yield_file)
    
    # Standardize
    df.rename(columns={df.columns[0]: 'Region'}, inplace=True)
    df_long = pd.melt(df, id_vars=['Region'], var_name='Year', value_name='Yield')
    
    # OPTIMIZATION: Vectorized String Cleaning (Faster than .apply)
    # Remove commas and quotes, convert to numeric
    df_long['Yield'] = pd.to_numeric(
        df_long['Yield'].astype(str).str.replace(',', '').str.replace('"', ''), 
        errors='coerce'
    )
    
    # Clean Year: Handle '2016-17' format
    def clean_year(x):
        try:
            parts = str(x).split('-')
            if len(parts) == 2: return int("20" + parts[1])
            return int(x)
        except: return None

    df_long['Year'] = df_long['Year'].apply(clean_year)
    
    # Filter
    df_clean = df_long.dropna(subset=['Yield', 'Year']).copy()
    df_clean['Year'] = df_clean['Year'].astype(int)
    
    # Filter Years & Region
    df_clean = df_clean[(df_clean['Year'] >= 2017) & (df_clean['Year'] <= 2020)]
    df_clean = df_clean[df_clean['Region'].str.strip().str.lower() == TARGET_REGION.lower()]
    
    print(f" -> Found {len(df_clean)} valid yield records for '{TARGET_REGION}'.")
    return df_clean

# %%
def process_weather_data(all_csvs):
    print("\n[2/3] Processing Weather Data...")
    weather_dfs = []
    
    for csv_file in all_csvs:
        # Skip non-weather files
        if any(x in csv_file for x in ["yield_data", "final_model", "dataset", "processed"]):
            continue
            
        try:
            # OPTIMIZATION: Read only first 30 lines to check header
            with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                head = [next(f) for _ in range(30)]
            
            # Check for keywords
            file_content = "".join(head)
            if not any(x in file_content for x in ["NASA", "POWER", "T2M", "YEAR,DOY"]):
                continue

            # Find start row
            skip_rows = 0
            for i, line in enumerate(head):
                if "YEAR,DOY" in line or ("YEAR" in line and "T2M" in line):
                    skip_rows = i
                    break
            
            df = pd.read_csv(csv_file, skiprows=skip_rows)
            
            # Column Mapping
            rename_map = {
                'T2M': 'temp', 'T2M_MAX': 'temp_max', 'T2M_MIN': 'temp_min',
                'PRECTOTCORR': 'precip', 'ALLSKY_SFC_SW_DWN': 'solar_rad'
            }
            # Rename available columns
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            
            if 'YEAR' in df.columns and 'temp' in df.columns:
                weather_dfs.append(df)
                print(f" -> ✅ Accepted: {os.path.basename(csv_file)}")
            else:
                # Debug info for rejected files
                print(f" -> ⚠️ Rejected {os.path.basename(csv_file)} (Missing columns). Found: {list(df.columns)}")
            
        except Exception: 
            pass

    if not weather_dfs:
        print("ERROR: No valid weather files found!")
        return None

    full_weather = pd.concat(weather_dfs, ignore_index=True)
    
    # Robust Date Creation
    try:
        # Ensure integers
        full_weather['YEAR'] = pd.to_numeric(full_weather['YEAR'], errors='coerce').fillna(0).astype(int)
        full_weather['DOY'] = pd.to_numeric(full_weather['DOY'], errors='coerce').fillna(0).astype(int)
        
        # Convert to datetime
        full_weather['Date'] = pd.to_datetime(
            full_weather['YEAR'].astype(str) + '-' + full_weather['DOY'].astype(str), 
            format="%Y-%j"
        )
        full_weather = full_weather.sort_values('Date').drop_duplicates(subset=['Date'])
        full_weather = full_weather.fillna(method='ffill')
        
        print(f" -> Weather Timeline: {full_weather['Date'].min().date()} to {full_weather['Date'].max().date()}")
        return full_weather
    except Exception as e:
        print(f"Error parsing dates: {e}")
        return None

# %%
def link_datasets(df_yield, df_weather):
    print("\n[3/3] Linking Datasets...")
    
    # OPTIMIZATION: Build Image Cache {Year: Path} once
    # This avoids O(N*M) looping later.
    print(" -> Indexing images...")
    image_year_map = {}
    
    # Recursive search for all extensions
    all_images = []
    for ext in ["*.tif", "*.jp2", "*.TIF", "*.tiff"]:
        all_images.extend(glob.glob(os.path.join(IMAGES_DIR, "**", ext), recursive=True))
    
    for img_path in all_images:
        filename = os.path.basename(img_path)
        # Check for years 2016-2026 in filename
        for y in range(2016, 2027):
            if str(y) in filename:
                # Store relative path
                if img_path.startswith(os.getcwd()):
                    img_path = os.path.relpath(img_path)
                image_year_map[y] = img_path
                break
                
    print(f" -> Indexed {len(image_year_map)} years of satellite imagery.")
    
    final_dataset = []
    weather_output_dir = os.path.join(OUTPUT_DIR, "weather_sequences")
    if not os.path.exists(weather_output_dir):
        os.makedirs(weather_output_dir)

    for index, row in df_yield.iterrows():
        year = int(row['Year'])
        region = row['Region']
        
        # 1. Weather Slicing
        harvest_date = datetime(year, SEASON_END_MONTH, 30)
        sowing_date = datetime(year - 1, SEASON_START_MONTH, 1)
        
        mask = (df_weather['Date'] >= sowing_date) & (df_weather['Date'] <= harvest_date)
        season_weather = df_weather.loc[mask].copy()
        
        if len(season_weather) < 100: 
            continue
            
        cols_to_save = ['temp', 'precip', 'solar_rad']
        for c in cols_to_save:
            if c not in season_weather.columns: season_weather[c] = 0.0
            
        weather_filename = f"{region}_{year}.csv"
        weather_path = os.path.join(weather_output_dir, weather_filename)
        season_weather[cols_to_save].to_csv(weather_path, index=False)
        
        # 2. Image Linking (Instant Lookup)
        actual_image_path = image_year_map.get(year, f"data/satellite_images/{region}_{year}.tif")
        
        final_dataset.append({
            "sample_id": f"{region}_{year}",
            "year": year,
            "region": region,
            "yield_label": row['Yield'],
            "weather_path": weather_path,
            "image_path": actual_image_path 
        })
        
    final_df = pd.DataFrame(final_dataset)
    save_path = os.path.join(OUTPUT_DIR, "final_model_dataset.csv")
    final_df.to_csv(save_path, index=False)
    print(f"\nSUCCESS: Generated final dataset with {len(final_df)} samples.")

# %%
if __name__ == "__main__":
    setup_directories()
    all_csvs = find_all_csvs()
    if all_csvs:
        yield_df = process_yield_data(all_csvs)
        if yield_df is not None:
            weather_df = process_weather_data(all_csvs)
            if weather_df is not None:
                link_datasets(yield_df, weather_df)