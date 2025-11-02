# Traffic Prediction with Graph Neural Networks

A Graph Neural Network approach for predicting travel times using real-time traffic and weather data.

## Project Overview
This project implements a GNN-based traffic prediction system for urban mobility analysis. The model learns spatio-temporal patterns from traffic sensor networks and weather conditions to forecast travel times.

## Project Structure
traffic_gnn/
├── src/ # Source code
├── data/ # Data directories
├── notebooks/ # Analysis & experiments
└── models/ # Trained models

## Getting Started
```bash
# 1. Download data
python src/download_data.py

# 2. Explore data (next step)
jupyter notebook notebooks/01_eda.ipynb

## Data Sources

### Traffic Data
- **Source**: IntraTraffic API
- **Features**: speed, congestion, coordinates, timestamps

### Weather Data
- **Source**: Open-Meteo Historical API
- **Features**: temperature, precipitation, timestamps
- **Coordinates**: 42.6629° N, 21.1655° E

### Coverage
- **Period**: March 2025 - October 2025
- **Location**: Prishtina, Kosovo

## Tech Stack
- **Python 3.10.12**
- **PyTorch & PyTorch Geometric** - Graph Neural Networks
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine learning utilities
- **Jupyter Notebooks** - Interactive exploration

## Project Status
- ** In Progress - Data collection complete, EDA phase starting