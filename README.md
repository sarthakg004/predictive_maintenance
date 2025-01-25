# TURBO: Predicting Remaining Useful Life (RUL) of Turbojet Engines

## Overview
TURBO is a Python-based project for predicting the Remaining Useful Life (RUL) of turbojet engines using both traditional machine learning and deep learning approaches. The project processes sensor data from engines, generates time-series sequences, and trains models such as a Generalized Linear Model (GLM) and a Deep Convolutional Neural Network (DCNN).

This repository provides tools for preprocessing, visualization, model training, and evaluation.

## Features
- **Data Preprocessing**: RUL calculation, operating condition encoding, feature scaling, and exponential smoothing.
- **Data Visualization**: Sensor trends and RUL distribution plots.
- **Traditional ML Model**: Generalized Linear Model (GLM) with SnapML's LinearRegression.
- **Deep Learning Model**: DCNN with convolutional and fully connected layers for regression.
- **Time-Series Preparation**: Sequence generation for engine sensor data.
- **Metrics**: Evaluate models using RMSE and R² scores.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tensorflow`
  - `snapml`
  - `scikit-learn`

Install the dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Initialize the Project
```python
from turbo import TURBO

# Initialize and preprocess data
turbo = TURBO()
```

### 2. Visualize Data
```python
# Visualize sensor trends and RUL distribution
turbo.plot()
```

### 3. Train a GLM Model
```python
# Train a Generalized Linear Model (GLM)
turbo.model_GLM()
```

### 4. Prepare Data for DCNN
```python
# Preprocess data for DCNN
turbo.prepare_data_DCNN()
```

### 5. Train a DCNN Model
```python
# Build and train the DCNN model
turbo.build_train_model(plot=True, epochs=20)
```

### 6. Evaluate the Model
```python
# Evaluate the trained DCNN
turbo.model_eval()
```

## File Structure
```
.
|-- turbo.py                # Main Python class for data processing and modeling
|-- requirements.txt        # List of dependencies
|-- data/
|   |-- train_FD001.txt     # Training data
|   |-- test_FD001.txt      # Testing data
|   |-- RUL_FD001.txt       # Actual RUL for testing
|-- README.md               # Project documentation
```

## Data Description
The dataset consists of sensor readings for multiple turbojet engines:
- **Training Data**: Time-series data up to engine failure.
- **Testing Data**: Time-series data up to an evaluation point.
- **RUL File**: Ground-truth RUL values for the testing dataset.

### Columns in the Dataset
- `unit_no`: Engine ID.
- `time_cycles`: Operational cycles.
- `op_setting_*`: Operating settings.
- `sensor_*`: Sensor readings.

## Model Architecture
The DCNN architecture includes:
1. **Convolutional Layers**: Extract spatial features from sensor data.
2. **Pooling Layers**: Downsample feature maps.
3. **Dropout Layers**: Prevent overfitting.
4. **Fully Connected Layers**: Combine extracted features for RUL prediction.

## Evaluation Metrics
- **Root Mean Squared Error (RMSE)**: Measures prediction accuracy.
- **R² Score**: Indicates model fit.

## Acknowledgments
The project uses the NASA C-MAPSS dataset for engine degradation simulations.
