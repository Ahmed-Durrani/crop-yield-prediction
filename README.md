**🌾 Predictive Crop Yield Modeling (CNN-LSTM)**

An automated Deep Learning system that predicts wheat crop yield using Satellite Imagery (Sentinel-2) and Historical Weather Data (NASA POWER).

**🚀 Features**

Hybrid Architecture: Combines CNN (for spatial image analysis) and LSTM (for temporal weather analysis).

Automated Pipeline: Single-click execution from data cleaning to visualization.

Scientific Loading: Supports 16-bit satellite imagery using tifffile and OpenCV.

Forecasting: Predicts future yields (2021-2025) based on real-time inputs.

Scenario Simulation: "What-If" analysis tool to simulate yields under different weather conditions (e.g., Drought vs. Flood).

**📂 Project Structure**

run_project.py: Start here. The master script that runs the entire pipeline.

clean_real_data.py: Preprocesses CSVs and aligns temporal data.

dataset_loader.py: Custom PyTorch loader with recursive search and random sampling.

model_architecture.py: Defines the Dual-Stream CNN-LSTM model.

train.py: Training loop with temporal train/test split.

inference.py: Evaluates performance on historical data.

future_dashboard.py: Visualizes predictions for 2017-2025.

**🛠️ Installation**

Clone the repository:

git clone [https://github.com/your-username/crop-yield-prediction.git](https://github.com/your-username/crop-yield-prediction.git)

cd crop-yield-prediction


Install dependencies:

pip install -r requirements.txt


Data Setup:

Place your .tif satellite images in data/satellite_images/.

Place your weather CSVs in the root folder or data/.

Ensure yield_data.csv is in the root folder.

**▶️ Usage**

Simply run the master script to execute the full pipeline:

python run_project.py


This will:

Clean and link the data.

Train the model (if enabled).

Generate an accuracy report.

Launch the Visual Dashboard.

**📊 Results**

Region: Multan, Pakistan

Test Accuracy: ~97.9%

MAE: 0.0113 Tonnes/Hectare
