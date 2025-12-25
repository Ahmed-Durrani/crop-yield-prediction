# %% [markdown]
# # Step 7: Combined Results Dashboard (2017-2025)
# This script visualizes inputs and PREDICTIONS for BOTH historical and future years.
# It displays them in a single, unified view.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import importlib.util
import os
import sys
import pandas as pd

# --- CONFIGURATION ---
MODEL_PATH = "models/best_checkpoint.pth"
HISTORICAL_CSV = "data/processed_real_data/final_model_dataset.csv"
FUTURE_CSV = "data/processed_real_data/future_dataset.csv"

# %%
def import_deps():
    """Dynamic import to reuse your existing files"""
    spec = importlib.util.spec_from_file_location("dataset_loader", "dataset_loader.py")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    
    spec = importlib.util.spec_from_file_location("model_architecture", "model_architecture.py")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    return loader_mod, model_mod

# %%
def visualize_dashboard():
    print(f"--- GENERATING UNIFIED DASHBOARD ---")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Trained model not found. Run train.py first.")
        return

    # 1. Load Modules & Data
    try:
        loader_mod, model_mod = import_deps()
    except FileNotFoundError:
        print("Error: Dependency files (dataset_loader.py, model_architecture.py) not found.")
        return

    CropYieldDataset = loader_mod.CropYieldDataset
    CropYieldHybridModel = model_mod.CropYieldHybridModel
    
    # Load Data
    ds_history = None
    if os.path.exists(HISTORICAL_CSV):
        ds_history = CropYieldDataset(HISTORICAL_CSV, is_training=False)
    
    ds_future = None
    if os.path.exists(FUTURE_CSV):
        ds_future = CropYieldDataset(FUTURE_CSV, is_training=False)
    
    # Combine datasets for visualization
    all_samples = []
    
    if ds_history:
        print(f"Loaded {len(ds_history)} historical samples.")
        for i in range(len(ds_history)):
            all_samples.append((ds_history[i], ds_history.data_frame.iloc[i], "Historical"))
            
    if ds_future:
        print(f"Loaded {len(ds_future)} future samples.")
        for i in range(len(ds_future)):
            all_samples.append((ds_future[i], ds_future.data_frame.iloc[i], "Future"))
            
    num_samples = len(all_samples)
    if num_samples == 0:
        print("Error: No data found to visualize.")
        return

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropYieldHybridModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("\nProcessing predictions and plots...")
    print(f"{'Year':<6} | {'Type':<12} | {'Predicted (T/ha)':<20}")
    print("-" * 45)

    with torch.no_grad():
        for i in range(num_samples):
            (img_tensor, weather_tensor, target_tensor), row, data_type = all_samples[i]
            year = row['year']
            
            # Run Prediction
            img_input = img_tensor.unsqueeze(0).to(device)       # Add batch dim
            weather_input = weather_tensor.unsqueeze(0).to(device)
            prediction = model(img_input, weather_input).item()
            
            print(f"{year:<6} | {data_type:<12} | {prediction:.4f}")

            # Get Actual if available
            actual_text = ""
            if data_type == "Historical":
                actual = target_tensor.item()
                actual_text = f" | Act: {actual:.3f}"
            
            # --- Prepare Image for Plotting ---
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            
            img_np = img_tensor.numpy()
            img_np = (img_np * std) + mean
            img_np = np.clip(img_np, 0, 1)
            img_np = np.transpose(img_np, (1, 2, 0)) 

            # --- Prepare Weather for Plotting ---
            weather_np = weather_tensor.numpy()
            temp = weather_np[:, 0] * 50
            rain = weather_np[:, 1] * 50
            
            # --- CREATE INDIVIDUAL FIGURE (Scrollable Feed) ---
            fig, (ax_img, ax_weather) = plt.subplots(1, 2, figsize=(12, 4))
            
            # --- Plot Left: Image ---
            ax_img.imshow(img_np)
            
            status = ""
            if "MISSING" in row['image_path']:
                status = "\n(⚠️ No Image)"
            
            title_color = "black" if data_type == "Historical" else "blue"
            title_text = f"{year} ({data_type}) Pred: {prediction:.3f}{actual_text} T/ha{status}"
            
            ax_img.set_title(title_text, fontsize=12, fontweight='bold', color=title_color)
            ax_img.axis('off')
            
            # --- Plot Right: Weather ---
            ax_weather.plot(temp, label="Temp", color="orange", linewidth=1.5)
            ax_weather.plot(rain, label="Rain", color="blue", linewidth=1.5, alpha=0.6)
            
            ax_weather.set_title(f"Weather Conditions ({year})", fontsize=10)
            ax_weather.set_ylabel("Value", fontsize=9)
            ax_weather.tick_params(axis='both', which='major', labelsize=8)
            ax_weather.legend(loc="upper right", fontsize=8)
            ax_weather.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show() # Show each year's block immediately

    print("\nDashboard generated successfully.")

if __name__ == "__main__":
    visualize_dashboard()