# %% [markdown]
# # Step 2: The Custom Dataset Loader (Optimized)
# This script defines the `CropYieldDataset` class. 
# OPTIMIZATIONS: 
# - Single-pass file grouping (O(N) instead of O(N*M)).
# - Tifffile/OpenCV support for scientific images.
# - Robust error handling and random sampling.

# %%
import os
# Silence OpenCV errors before import
os.environ["OPENCV_LOG_LEVEL"] = "OFF" 

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import glob
import cv2 

# Try to import tifffile (Best for satellite data)
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("⚠️ Warning: 'tifffile' not found. Run: pip install tifffile imagecodecs")

# Safety Overrides
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# Configuration
PROJECT_ROOT = "." 

# %%
class CropYieldDataset(Dataset):
    def __init__(self, csv_file, root_dir=PROJECT_ROOT, transform=None, is_training=True):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        self.weather_max_vals = np.array([50.0, 50.0, 40.0]) 
        
        # --- 1. FIND ALL IMAGES RECURSIVELY ---
        print("\n[Loader] Scanning for images...")
        patterns = [
            os.path.join(root_dir, "**", "*.tif"),
            os.path.join(root_dir, "**", "*.TIF"),
            os.path.join(root_dir, "**", "*.tiff"),
            os.path.join(root_dir, "**", "*.jp2")
        ]
        
        self.available_tifs = []
        for p in patterns:
            found = glob.glob(p, recursive=True)
            self.available_tifs.extend(found)
        
        self.available_tifs = list(set(self.available_tifs))
        print(f"[Loader] Found {len(self.available_tifs)} total image files.")

        # --- 2. GROUP IMAGES BY YEAR (OPTIMIZED) ---
        # Create a dictionary for fast lookups
        unique_years = sorted(self.data_frame['year'].unique())
        self.year_files = {year: [] for year in unique_years}
        
        # OPTIMIZATION: Iterate through files ONCE, not for every year
        print("-" * 50)
        print(f"{'Year':<6} | {'Status':<15} | {'Count':<10}")
        print("-" * 50)

        # Pre-compute string versions of years to avoid repeated casting
        year_map = {str(y): y for y in unique_years}

        for f in self.available_tifs:
            fname = os.path.basename(f)
            # Check which year this file belongs to
            for year_str, year_int in year_map.items():
                if year_str in fname:
                    self.year_files[year_int].append(f)
                    break # Stop checking other years for this file

        # Print Summary
        for year in unique_years:
            count = len(self.year_files[year])
            if count > 0:
                print(f"{year:<6} | ✅ Ready        | {count:<10}")
            else:
                print(f"{year:<6} | ❌ MISSING      | 0")
        print("-" * 50 + "\n")

    def __len__(self):
        return len(self.data_frame)

    def load_image_robust(self, path):
        """Tries Tifffile -> PIL -> OpenCV (Optimized order)"""
        
        # Strategy 1: Tifffile (Fastest for Scientific Data)
        if HAS_TIFFFILE:
            try:
                img_np = tifffile.imread(path)
                # Handle Sentinel dimensions (Band, Height, Width)
                if len(img_np.shape) == 3 and img_np.shape[0] > 4: 
                    img_np = img_np[:3, :, :] # Take first 3 bands
                    img_np = np.transpose(img_np, (1, 2, 0)) # CHW -> HWC
                elif len(img_np.shape) == 2:
                    img_np = np.stack((img_np,)*3, axis=-1) # Grayscale -> RGB
                elif len(img_np.shape) == 3 and img_np.shape[2] > 3:
                    img_np = img_np[:, :, :3] # Trim extra channels

                # Normalize 16-bit to 8-bit
                if img_np.max() > 255:
                    img_np = (img_np / img_np.max() * 255).astype('uint8')
                else:
                    img_np = img_np.astype('uint8')
                return Image.fromarray(img_np)
            except: pass

        # Strategy 2: PIL (Standard)
        try: return Image.open(path).convert('RGB')
        except: pass
        
        # Strategy 3: OpenCV (Fallback for complex headers)
        try:
            cv_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if cv_img is not None:
                if cv_img.dtype == 'uint16': 
                    cv_img = (cv_img / 256).astype('uint8')
                
                # Convert colorspace
                if len(cv_img.shape) == 2: 
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                elif len(cv_img.shape) == 3:
                    if cv_img.shape[2] == 3: 
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    elif cv_img.shape[2] == 4: 
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
                return Image.fromarray(cv_img)
        except: pass
        
        return None

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.data_frame.iloc[idx]

        # 1. Load Image (Random Sampling)
        year = row['year']
        potential_files = self.year_files.get(year, [])
        image = None
        attempts = 0
        
        # Try up to 5 random files if some are corrupt
        while image is None and len(potential_files) > 0 and attempts < 5:
            selected_path = str(np.random.choice(potential_files))
            image = self.load_image_robust(selected_path)
            
            if image is None:
                # Remove bad file from memory cache so we don't pick it again
                if selected_path in potential_files:
                    potential_files.remove(selected_path)
            attempts += 1
        
        # Fallback
        if image is None: 
            image = Image.new('RGB', (64, 64), (0, 0, 0))

        if self.transform: 
            image = self.transform(image)
        else:
            default_tf = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = default_tf(image)

        # 2. Load Weather
        weather_path = row['weather_path']
        if not os.path.isabs(weather_path): 
            weather_path = os.path.join(self.root_dir, weather_path)
            
        try:
            weather_df = pd.read_csv(weather_path)
            weather_vals = weather_df[['temp', 'precip', 'solar_rad']].values
            weather_vals = weather_vals / self.weather_max_vals
        except: 
            weather_vals = np.zeros((180, 3))
        
        # Pad/Truncate to 180 days
        target_len = 180
        curr_len = weather_vals.shape[0]
        if curr_len < target_len:
            padding = np.zeros((target_len - curr_len, 3))
            weather_vals = np.vstack([weather_vals, padding])
        elif curr_len > target_len:
            weather_vals = weather_vals[:target_len, :]
            
        weather_tensor = torch.tensor(weather_vals, dtype=torch.float32)

        # 3. Load Yield (Scaled: kg -> Tonnes)
        yield_val = float(row['yield_label']) / 1000.0 
        yield_tensor = torch.tensor(yield_val, dtype=torch.float32)

        return image, weather_tensor, yield_tensor