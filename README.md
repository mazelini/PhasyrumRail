# PhysarumRail: Optimizing Railway Networks with Machine Learning and Bio-Inspired Reinforcement

## Overview
PhysarumRail is a tool that uses machine learning and slime mold simulation algorithms to optimize railway networks. The system analyzes geographic data to suggest optimal railway routes between specified locations.

## Installation

1. Create a virtual environment (recommended)
2. Activate the virtual environment
3. Install dependencies using `pip install -r requirements.txt`

## Usage

### Running the Streamlit Web Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Access the web interface in your browser (typically http://localhost:8501)

3. In the application:
   - Select geographic locations for origin and destination
   - Set parameters for the railway network optimization
   - Run the simulation to generate optimal routes
   - View and export results

### Training the Model

To retrain the machine learning model with new data:

```
python train_model.py
```

### Extracting Features for Training

To extract features from geographic data for model training:

```
python extract_features_for_training.py
```

## Project Structure

- `app.py` - Main Streamlit web application
- `train_model.py` - Script for training the machine learning model
- `extract_features_for_training.py` - Extracts features for model training
- `utils/` - Utility functions
- `sim/` - Simulation code implementing slime mold algorithm for network optimization
- `models/` - Stores trained machine learning models
- `data/` - Geographic data and datasets
- `results/` - Output files and generated maps

## Data Files

- **Elevation Rasters**: Located in `data/elevation_rasters/` directory
- **Population Rasters**: Located in `data/population_rasters/` directory
- **Network Graphs**: Located in `data/graphs/` directory
- **Training Data**: The main training dataset is `data/ai_training_edges_full_enhanced.csv`

## Notes

- For large geographic areas, processing may take significant time
- The results directory contains example output maps from previous runs
- GPU acceleration is supported for certain operations if available
