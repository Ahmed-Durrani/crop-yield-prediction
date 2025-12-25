# %% [markdown]
# # Project Master Runner (Optimized)
# Executes the entire pipeline INLINE using runpy.
# Use the flags below to skip steps for speed!

# %%
import runpy
import os
import sys
import matplotlib.pyplot as plt

# --- OPTIMIZATION FLAGS ---
# Set to False to skip steps you have already completed.
ENABLE_CLEANING = True       # Re-scans files and builds CSVs
ENABLE_TRAINING = False       # Trains the model (Set False to use saved brain)
ENABLE_INFERENCE = True      # Prints Historical Accuracy (2017-2020)
ENABLE_FUTURE_PRED = True    # Generates predictions for 2021-2025
ENABLE_VISUALIZATION = True  # Shows the Final Graph/Image Dashboard
ENABLE_SCENARIO_SIM = True   # Runs the 2026 Scenario Simulation

def run_pipeline():
    print("="*50)
    print("   🌱 CROP YIELD AI: AUTOMATED PIPELINE")
    print("="*50)

    # Define Pipeline: (Flag, Step Name, File Name)
    steps = [
        (ENABLE_CLEANING, "Step 1: Data Cleaning", "clean_real_data.py"),
        (ENABLE_TRAINING, "Step 2: Model Training", "train.py"),
        (ENABLE_INFERENCE, "Step 3: Historical Report", "inference.py"),
        (ENABLE_FUTURE_PRED, "Step 4: Future Prediction", "predict_future.py"),
        (ENABLE_VISUALIZATION, "Step 5: Final Dashboard", "future_dashboard.py"),
        (ENABLE_SCENARIO_SIM, "Step 6: Scenario Simulation", "simulate_scenario.py")
    ]

    for should_run, step_name, script_name in steps:
        if not should_run:
            print(f"⏩ Skipping {step_name} (Flag set to False)")
            continue

        if not os.path.exists(script_name):
            print(f"⚠️  Warning: '{script_name}' not found. Skipping.")
            continue

        print(f"\n>>> Running {step_name}...")
        try:
            # Run inline to keep variables/plots in one window
            runpy.run_path(script_name, run_name="__main__")
            print(f"✅ {step_name} Complete.")
        except Exception as e:
            print(f"❌ Critical Error in {script_name}: {e}")
            # If training fails, the rest will likely fail, so we stop.
            if "train" in script_name: 
                print("Stopping pipeline due to training failure.")
                return 

    print("\n" + "="*50)
    print("       🎉 PIPELINE EXECUTION FINISHED")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()
# %%
