# %% [markdown]
# # Step 8: Scenario Simulation Dashboard ("The Multiverse")
# This script generates 3 random "What-If" scenarios for 2026.
# It visualizes the inputs (borrowed from history) and predicts the outcome.
# It benchmarks the result against the historical average to determine impact.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
import shutil
import torch
from torch.utils.data import DataLoader
import importlib.util
import random

# --- CONFIGURATION ---
TARGET_YEAR = 2026
REGION = "Multan"
NUM_SCENARIOS = 3
AVAILABLE_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Paths
MODEL_PATH = "models/best_checkpoint.pth"
DATA_DIR = "data/processed_real_data"
WEATHER_DIR = os.path.join(DATA_DIR, "weather_sequences")
OUTPUT_CSV = os.path.join(DATA_DIR, "scenario_dataset.csv")
HISTORICAL_CSV = os.path.join(DATA_DIR, "final_model_dataset.csv")

# %%
def import_deps():
    spec = importlib.util.spec_from_file_location("dataset_loader", "dataset_loader.py")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    
    spec = importlib.util.spec_from_file_location("model_architecture", "model_architecture.py")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    return loader_mod, model_mod

# %%
def get_historical_average():
    if os.path.exists(HISTORICAL_CSV):
        df = pd.read_csv(HISTORICAL_CSV)
        # Yield is in kg in CSV, but model predicts Tonnes (kg/1000)
        # Let's convert CSV average to Tonnes for comparison
        return df['yield_label'].mean() / 1000.0
    return 0.600 # Fallback default

# %%
def create_multiverse_scenarios():
    print(f"\n--- 🌌 GENERATING {NUM_SCENARIOS} PARALLEL UNIVERSES FOR {TARGET_YEAR} ---")
    
    scenario_list = []
    
    # Find all available image files once
    image_files = glob.glob("data/satellite_images/**/*.tif", recursive=True) + \
                  glob.glob("data/satellite_images/**/*.jp2", recursive=True)

    for i in range(NUM_SCENARIOS):
        # 1. Pick Random Ingredients
        w_year = random.choice(AVAILABLE_YEARS)
        i_year = random.choice(AVAILABLE_YEARS)
        
        scenario_id = f"Sim_{i+1}"
        print(f"   🔮 {scenario_id}: Weather({w_year}) + Crop({i_year})")
        
        # 2. Prepare Weather
        source_weather_path = os.path.join(WEATHER_DIR, f"{REGION}_{w_year}.csv")
        if not os.path.exists(source_weather_path):
            source_weather_path = os.path.join(WEATHER_DIR, f"FUTURE_{REGION}_{w_year}.csv")
            
        if not os.path.exists(source_weather_path):
            print(f"      ⚠️ Weather file missing for {w_year}, skipping...")
            continue
            
        target_weather_name = f"SCENARIO_{i}_{TARGET_YEAR}.csv"
        target_weather_path = os.path.join(WEATHER_DIR, target_weather_name)
        shutil.copy(source_weather_path, target_weather_path)
        
        # 3. Prepare Image
        target_image_path = "MISSING"
        for img in image_files:
            if str(i_year) in os.path.basename(img):
                target_image_path = img
                if target_image_path.startswith(os.getcwd()):
                    target_image_path = os.path.relpath(target_image_path)
                break
        
        # 4. Add to list
        scenario_list.append({
            "sample_id": scenario_id,
            "year": TARGET_YEAR,
            "region": REGION,
            "yield_label": 0.0,
            "weather_path": target_weather_path,
            "image_path": target_image_path,
            "meta_weather_src": w_year,
            "meta_image_src": i_year
        })
    
    df = pd.DataFrame(scenario_list)
    df.to_csv(OUTPUT_CSV, index=False)
    return OUTPUT_CSV, scenario_list

# %%
def run_simulation():
    res = create_multiverse_scenarios()
    if not res: return
    csv_path, meta_data = res

    loader_mod, model_mod = import_deps()
    CropYieldDataset = loader_mod.CropYieldDataset
    CropYieldHybridModel = model_mod.CropYieldHybridModel
    
    # Load Data
    dataset = CropYieldDataset(csv_file=csv_path, is_training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropYieldHybridModel().to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else: return
    model.eval()
    
    # Historical Baseline
    hist_avg = get_historical_average()
    print(f"\n📊 Historical Average Benchmark: {hist_avg:.4f} T/ha")

    # Setup Visualization
    fig, axes = plt.subplots(len(dataset), 2, figsize=(12, 4 * len(dataset)))
    if len(dataset) == 1: axes = [axes]
    
    print("\n--- SIMULATION RESULTS ---")
    
    with torch.no_grad():
        for i, (images, weather, _) in enumerate(dataloader):
            # Prediction
            img_input = images.to(device)
            weather_input = weather.to(device)
            prediction = model(img_input, weather_input).item()
            
            # Metadata
            meta = meta_data[i]
            w_src = meta['meta_weather_src']
            i_src = meta['meta_image_src']
            
            # Analysis
            diff = prediction - hist_avg
            status = "🟢 ABOVE AVG" if diff > 0 else "🔴 BELOW AVG"
            pct_diff = (diff / hist_avg) * 100
            
            print(f"Scenario {i+1}: Pred {prediction:.4f} T/ha | {status} ({pct_diff:+.1f}%)")

            # --- PLOTTING ---
            # 1. Image
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_np = images[0].numpy()
            img_np = (img_np * std) + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = np.transpose(img_np, (1, 2, 0))
            
            ax_img = axes[i][0]
            ax_img.imshow(img_np)
            ax_img.set_title(f"Scenario {i+1}: Visuals from {i_src}\nPred: {prediction:.3f} T/ha", fontsize=12, fontweight='bold')
            ax_img.axis('off')
            
            # 2. Weather
            weather_np = weather[0].numpy()
            temp = weather_np[:, 0] * 50
            rain = weather_np[:, 1] * 50
            
            ax_weather = axes[i][1]
            ax_weather.plot(temp, label=f"Temp ({w_src})", color="orange", linewidth=1.5)
            ax_weather.plot(rain, label=f"Rain ({w_src})", color="blue", linewidth=1.5, alpha=0.6)
            
            # Add benchmark line
            ax_weather.set_title(f"Weather Conditions (Source: {w_src})\nImpact: {status}", fontsize=10)
            ax_weather.legend(loc="upper right", fontsize=8)
            ax_weather.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()
# %%
