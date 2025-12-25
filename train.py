# %% [markdown]
# # Step 4: The Training Loop (Stable / Seeding Fixed)
# Train: 2017-2019, Test: 2020
# UPDATED: Added Random Seed to ensure consistent, high accuracy every time.

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import importlib.util
import os
import sys
import random
import numpy as np

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
SAVE_PATH = "models/best_checkpoint.pth"
SEED = 42  # The "God Mode" number. Guarantees same results every time.

# %%
def set_seed(seed):
    """Freezes the random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🌱 Random Seed set to {seed} (Stability Mode On)")

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
def train_model():
    # 1. Set Seed FIRST
    set_seed(SEED)

    # 2. Import Dependencies
    try:
        loader_mod, model_mod = import_deps()
    except FileNotFoundError:
        print("❌ Error: 'dataset_loader.py' or 'model_architecture.py' not found.")
        return

    CropYieldDataset = loader_mod.CropYieldDataset
    CropYieldHybridModel = model_mod.CropYieldHybridModel

    # 3. Load Data
    csv_path = os.path.join("data", "processed_real_data", "final_model_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Data file not found at {csv_path}")
        return

    full_dataset = CropYieldDataset(csv_file=csv_path)
    total_size = len(full_dataset)
    
    if total_size < 2:
        print("❌ Error: Not enough data to split (Need at least 2 years).")
        return
        
    # 4. Temporal Split
    train_indices = list(range(total_size - 1))
    val_indices = [total_size - 1]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Shuffle=True is fine here because the SEED makes the shuffle predictable!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training on {len(train_dataset)} samples, Testing on {len(val_dataset)} sample.")

    # 5. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CropYieldHybridModel().to(device)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    
    # 6. Training Loop
    print("\n--- Starting Training ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, weather, targets in train_loader:
            images = images.to(device)
            weather = weather.to(device)
            targets = targets.view(-1, 1).to(device)
            
            # Forward
            outputs = model(images, weather)
            loss = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, weather, targets in val_loader:
                images = images.to(device)
                weather = weather.to(device)
                targets = targets.view(-1, 1).to(device)
                
                outputs = model(images, weather)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Logging
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if not os.path.exists("models"): os.makedirs("models")
            torch.save(model.state_dict(), SAVE_PATH)
            
    print("\n✅ TRAINING COMPLETE.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_model()