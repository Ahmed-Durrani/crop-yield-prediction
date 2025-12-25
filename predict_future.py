# %% [markdown]
# # Step 6: Future Year Predictor (2021-2025)
# RECURSIVE FIX: Scans all folders for weather files.

# %%
import pandas as pd
import numpy as np
import os
import glob
import torch
from torch.utils.data import DataLoader
import importlib.util
from datetime import datetime

TARGET_YEARS = [2021, 2022, 2023, 2024, 2025]
REGION = "Multan"
MODEL_PATH = "models/best_checkpoint.pth"
OUTPUT_DIR = "data/processed_real_data"
SEASON_START_MONTH = 11 
SEASON_END_MONTH = 4

def import_deps():
    spec = importlib.util.spec_from_file_location("dataset_loader", "dataset_loader.py")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    spec = importlib.util.spec_from_file_location("model_architecture", "model_architecture.py")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    return loader_mod, model_mod

def prepare_future_data():
    print(f"--- PREPARING FUTURE DATA ({TARGET_YEARS[0]}-{TARGET_YEARS[-1]}) ---")
    
    # RECURSIVE SEARCH for Weather CSVs
    weather_files = glob.glob("**/*.csv", recursive=True)
    weather_files = [f for f in weather_files if "dataset" not in f and "FUTURE" not in f]
    
    weather_dfs = []
    print(f"Scanning {len(weather_files)} CSV files for weather data...")
    
    for f in weather_files:
        try:
            with open(f, 'r') as file: head = [next(file) for _ in range(30)]
            skip, is_valid = 0, False
            for i, line in enumerate(head):
                if "YEAR,DOY" in line or ("YEAR" in line and "T2M" in line): 
                    skip = i; is_valid = True; break
            
            if is_valid:
                df = pd.read_csv(f, skiprows=skip)
                rename_map = {'T2M': 'temp', 'PRECTOTCORR': 'precip', 'ALLSKY_SFC_SW_DWN': 'solar_rad'}
                found_cols = {k: v for k, v in rename_map.items() if k in df.columns}
                df.rename(columns=found_cols, inplace=True)
                
                if 'YEAR' in df.columns and 'temp' in df.columns:
                    weather_dfs.append(df)
                    print(f" -> ✅ Loaded: {os.path.basename(f)}")
        except: pass
        
    if not weather_dfs:
        print("\n❌ ERROR: No valid weather files found! Download NASA POWER CSVs for 2021-2025.")
        return None

    full_weather = pd.concat(weather_dfs, ignore_index=True)
    try:
        full_weather['Date'] = pd.to_datetime(full_weather['YEAR'].astype(int).astype(str) + '-' + full_weather['DOY'].astype(int).astype(str), format="%Y-%j")
    except: return None
        
    full_weather = full_weather.sort_values('Date').drop_duplicates('Date').fillna(method='ffill')
    
    # Build Dataset
    future_data = []
    weather_out_dir = os.path.join(OUTPUT_DIR, "weather_sequences")
    if not os.path.exists(weather_out_dir): os.makedirs(weather_out_dir)
    
    image_files = glob.glob("data/satellite_images/**/*.tif", recursive=True) + glob.glob("data/satellite_images/**/*.jp2", recursive=True)
    
    print("\n--- MATCHING INPUTS ---")
    for year in TARGET_YEARS:
        start_date = datetime(year-1, SEASON_START_MONTH, 1)
        end_date = datetime(year, SEASON_END_MONTH, 30)
        mask = (full_weather['Date'] >= start_date) & (full_weather['Date'] <= end_date)
        season_weather = full_weather.loc[mask].copy()
        
        weather_status, weather_path = "❌ Missing", ""
        if len(season_weather) > 100:
            cols = ['temp', 'precip', 'solar_rad']
            for c in cols: 
                if c not in season_weather.columns: season_weather[c] = 0.0
            fname = f"FUTURE_{REGION}_{year}.csv"
            weather_path = os.path.join(weather_out_dir, fname)
            season_weather[cols].to_csv(weather_path, index=False)
            weather_status = "✅ Ready"
            
        image_path, image_status = "MISSING", "❌ Missing"
        for img in image_files:
            if str(year) in os.path.basename(img):
                image_path = img; image_status = "✅ Found"; break
                
        print(f"Year {year}: Weather={weather_status} | Image={image_status}")
        
        if weather_status == "✅ Ready": 
            future_data.append({
                "sample_id": f"FUTURE_{year}", "year": year, "region": REGION,
                "yield_label": 0.0, "weather_path": weather_path, "image_path": image_path
            })
            
    if not future_data: return None
    df_future = pd.DataFrame(future_data)
    save_path = os.path.join(OUTPUT_DIR, "future_dataset.csv")
    df_future.to_csv(save_path, index=False)
    return save_path

def run_prediction():
    csv_path = prepare_future_data()
    if not csv_path: return
    loader_mod, model_mod = import_deps()
    dataset = loader_mod.CropYieldDataset(csv_file=csv_path, is_training=False)
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_mod.CropYieldHybridModel().to(device)
    if os.path.exists(MODEL_PATH): model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else: return
    model.eval()
    
    print("\n" + "="*60 + f"\n   🔮 FUTURE YIELD PREDICTIONS ({REGION})   \n" + "="*60)
    print(f"{'Year':<6} | {'Predicted Yield (Tonnes/ha)':<30}\n" + "-" * 60)
    
    with torch.no_grad():
        for i, (images, weather, _) in enumerate(dataloader):
            outputs = model(images.to(device), weather.to(device))
            row = dataset.data_frame.iloc[i]
            note = "(⚠️ No Image)" if "MISSING" in row['image_path'] else ""
            print(f"{row['year']:<6} | {outputs.item():.4f} {note}")

if __name__ == "__main__":
    run_prediction()