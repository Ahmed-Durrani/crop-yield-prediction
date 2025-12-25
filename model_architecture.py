# %% [markdown]
# # Step 3: The Hybrid CNN-LSTM Architecture (Optimized)
# Structure:
# - CNN Branch: Processes Satellite Images (Spatial)
# - LSTM Branch: Processes Weather Logs (Temporal)
# - Fusion Head: Combines both to predict Yield

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration
IMG_SIZE = 64
NUM_WEATHER_FEATURES = 3
LSTM_HIDDEN_DIM = 64
CNN_OUT_DIM = 128

# %%
class CNN_Branch(nn.Module):
    def __init__(self):
        super(CNN_Branch, self).__init__()
        
        # Optimized: Grouping layers into a Sequential block
        self.conv_block = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Flatten Dimension: 64 channels * 8 height * 8 width
        self.flatten_dim = 64 * 8 * 8 
        
        # Feature Extractor Head
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, CNN_OUT_DIM),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # 1. Extract Spatial Features
        x = self.conv_block(x)
        
        # 2. Flatten (Batch, Channel, H, W) -> (Batch, Features)
        x = x.view(x.size(0), -1) 
        
        # 3. Compress to Feature Vector
        x = self.fc(x)
        return x

class LSTM_Branch(nn.Module):
    def __init__(self):
        super(LSTM_Branch, self).__init__()
        self.lstm = nn.LSTM(
            input_size=NUM_WEATHER_FEATURES, 
            hidden_size=LSTM_HIDDEN_DIM, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
    def forward(self, x):
        # LSTM output: (out, (hidden_state, cell_state))
        # We only need the hidden state of the LAST time step
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (num_layers, batch, hidden_size)
        # Take the last layer's hidden state
        return h_n[-1, :, :]

class CropYieldHybridModel(nn.Module):
    def __init__(self):
        super(CropYieldHybridModel, self).__init__()
        self.cnn = CNN_Branch()
        self.lstm = LSTM_Branch()
        
        # Fusion Dimension
        self.fusion_dim = CNN_OUT_DIM + LSTM_HIDDEN_DIM
        
        # Final Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(self.fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output 1 continuous value (Yield)
        )

    def forward(self, image, weather):
        # 1. Process Inputs separately
        cnn_out = self.cnn(image)
        lstm_out = self.lstm(weather)
        
        # 2. Fuse (Concatenate)
        combined = torch.cat((cnn_out, lstm_out), dim=1)
        
        # 3. Predict
        output = self.regressor(combined)
        return output