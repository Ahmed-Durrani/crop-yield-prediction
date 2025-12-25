# %% [markdown]
# # Step 5: Inference (Prediction Test)

# %%
import torch
from torch.utils.data import DataLoader
import importlib.util
import os
import sys

MODEL_PATH = "models/best_checkpoint.pth"
DATA_CSV = "data/processed_real_data/final_model_dataset.csv"

def import_deps():
    spec = importlib.util.spec_from_file_location("dataset_loader", "dataset_loader.py")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    
    spec = importlib.util.spec_from_file_location("model_architecture", "model_architecture.py")
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    return loader_mod, model_mod

def predict():
    # Dynamic Import
    try:
        loader_mod, model_mod = import_deps()
    except FileNotFoundError:
        print("Error: Dependencies not found.")
        return

    CropYieldDataset = loader_mod.CropYieldDataset
    CropYieldHybridModel = model_mod.CropYieldHybridModel
    
    if not os.path.exists(MODEL_PATH):
        print("Error: No model found. Run train.py first.")
        return

    if not os.path.exists(DATA_CSV):
        print(f"Error: Data file not found at {DATA_CSV}")
        return

    dataset = CropYieldDataset(csv_file=DATA_CSV, is_training=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropYieldHybridModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    print("\n" + "="*70)
    print(f"{'Year':<6} | {'Region':<12} | {'Actual Yield':<15} | {'Predicted':<15} | {'Diff'}")
    print("="*70)
    
    total_error = 0.0
    avg_yield = dataset.data_frame['yield_label'].mean() / 1000.0
    
    with torch.no_grad():
        for i, (images, weather, targets) in enumerate(dataloader):
            images, weather = images.to(device), weather.to(device)
            outputs = model(images, weather)
            predicted = outputs.item()
            actual = targets.item()
            
            row = dataset.data_frame.iloc[i]
            diff = abs(predicted - actual)
            total_error += diff
            
            print(f"{row['year']:<6} | {row['region']:<12} | {actual:.4f} T/ha     | {predicted:.4f} T/ha     | {diff:.4f}")

    print("="*70)
    if len(dataset) > 0:
        avg_error = total_error / len(dataset)
        acc = (1 - (avg_error / avg_yield)) * 100
        print(f"Average Error: {avg_error:.4f} T/ha | Accuracy: ~{acc:.1f}%")
    else:
        print("Dataset is empty.")

if __name__ == "__main__":
    predict()