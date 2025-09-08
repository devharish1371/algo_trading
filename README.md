# 🚀 Algo Trading with DCRNN (Diffusion Graph Convolutional Recurrent Neural Network)

A sophisticated algorithmic trading system that uses Deep Graph Convolutional Recurrent Neural Networks (DCRNN) to predict stock prices by leveraging both temporal patterns and inter-stock relationships in the market.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Data Requirements](#data-requirements)
- [Training & Inference](#training--inference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a state-of-the-art deep learning approach for algorithmic trading that combines:

- **Graph Neural Networks**: Captures relationships between different stocks
- **Recurrent Neural Networks**: Learns temporal patterns in price movements
- **Multi-stock Modeling**: Simultaneously predicts prices for multiple stocks
- **Graph Convolution**: Leverages market structure information for better predictions

The system is designed to predict future stock prices based on historical data and the relationships between different stocks in the market, making it ideal for portfolio optimization and trading strategy development.

## 🏗️ Architecture

### Core Components

1. **DCRNN Model** (`dcrnn_model.py`)
   - Encoder-Decoder architecture with attention mechanisms
   - Graph convolution layers for capturing inter-stock relationships
   - Bidirectional processing for temporal dependencies

2. **Graph Convolution Cell** (`dcrnn_cell.py`)
   - Custom GRU cell with graph convolution operations
   - Multiple filter types: Laplacian, Random Walk, Dual Random Walk
   - Configurable diffusion steps for graph propagation

3. **Training Supervisor** (`dcrnn_supervisor.py`)
   - Manages training loop and model checkpointing
   - Implements early stopping and learning rate scheduling
   - Handles validation and testing phases

4. **Data Pipeline** (`utils.py`)
   - Time series windowing and preprocessing
   - Graph construction from correlation matrices
   - Data scaling and normalization

### Data Flow

```
Raw Stock Data → Preprocessing → Graph Construction → DCRNN Model → Price Predictions
     ↓              ↓              ↓              ↓           ↓
  CSV/Parquet → Time Windows → Adjacency Matrix → Training → Inference
```

## ✨ Features

- **🔄 Multi-step Forecasting**: Predict multiple time steps ahead
- **📊 Graph-based Learning**: Captures market structure and correlations
- **🎯 Flexible Input**: Supports various stock data formats
- **🚀 GPU Acceleration**: CUDA support for faster training
- **📈 Real-time Ready**: Can be integrated with live trading systems
- **🔧 Configurable**: Easy parameter tuning for different markets
- **📝 Comprehensive Logging**: TensorBoard integration for monitoring
- **🌐 Colab Compatible**: Easy integration with Google Colab
- **💼 Automated Trading Strategy**: Generate portfolio allocation and recommendations
- **📊 Risk Management**: Threshold-based stock selection and filtering
- **💰 Portfolio Optimization**: Equal-weight allocation with performance metrics

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- 8GB+ RAM recommended

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd algo_trading
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## 🚀 Quick Start

### 1. Complete Demo with Trading Strategy

```bash
# Create sample data and run complete demo
python3 quick_demo.py --create-data
python3 quick_demo.py

# This will:
# 1. Train the DCRNN model
# 2. Generate price predictions
# 3. Create trading strategy with portfolio allocation
# 4. Save results to CSV files
```

### 2. Use Real Market Data

```bash
# Fetch BSE500 data
python3 data_fetch.py

# Train the model
python3 train.py --data_csv bse500_data.csv --epochs 30
```

### 3. Generate Predictions & Trading Strategy

```bash
# Use trained model for predictions and strategy
python3 trading_strategy.py --checkpoint logs/checkpoints/model_epoch_29.pt \
                           --recent_csv recent_data.csv \
                           --out_csv predictions.csv

# Or use the complete pipeline from quick_demo.py
# This automatically generates trading strategy with portfolio allocation
```

## 📚 Usage Examples

### Command Line Training

```bash
# Basic training
python3 train.py --data_csv stock_data.csv --epochs 50

# Advanced training with custom parameters
python3 train.py \
    --data_csv stock_data.csv \
    --seq_len 24 \
    --horizon 5 \
    --batch_size 64 \
    --epochs 100 \
    --base_lr 0.001 \
    --log_dir custom_logs
```

### Python API Usage

```python
from train import train_colab

# Train with custom parameters
train_colab(
    data_csv='stock_data.csv',
    seq_len=24,
    horizon=5,
    batch_size=64,
    epochs=50,
    base_lr=0.001,
    log_dir='custom_logs'
)
```

### Google Colab Integration

```python
# In Colab notebook
!pip install -r requirements.txt
from train import train_colab

# Upload your data and train
train_colab(
    data_csv='/content/stock_data.csv',
    epochs=30,
    batch_size=32
)
```

### Trading Strategy Generation

```python
from quick_demo import generate_trading_strategy
import pandas as pd

# Example predictions DataFrame (horizon x tickers)
predictions = pd.DataFrame({
    'RELIANCE.BO': [0.008, 0.006, 0.004],
    'TCS.BO': [0.012, 0.009, 0.007],
    'INFY.BO': [0.015, 0.011, 0.008]
}, index=['Next_Day', 'Day_2', 'Day_3'])

# Generate trading strategy
strategy = generate_trading_strategy(
    predictions, 
    capital=100000,      # $100k capital
    threshold=0.005,     # 0.5% minimum return
    top_n=3              # Top 3 stocks
)

# Strategy includes:
# - Stock selection based on threshold
# - Portfolio allocation
# - Expected returns and profit
# - Trading recommendations
```

#### Customizing Trading Strategy

```python
# Adjust risk tolerance
strategy = generate_trading_strategy(
    predictions, 
    capital=50000,       # Lower capital
    threshold=0.01,      # Higher threshold (1% minimum return)
    top_n=5              # More diversified portfolio
)

# Conservative strategy
strategy = generate_trading_strategy(
    predictions, 
    capital=200000,      # Higher capital
    threshold=0.02,      # Very high threshold (2% minimum return)
    top_n=2              # Concentrated portfolio
)
```

## 📁 Project Structure

```
algo_trading/
├── 📁 venv/                    # Virtual environment
├── 📁 logs/                    # Training logs and checkpoints
├── 📁 demo_logs/               # Demo training outputs
├── 📁 test/                    # Test files
├── 📄 README.md                # This file
├── 📄 requirements.txt         # Python dependencies
├── 📄 quick_demo.py            # Complete demo with trading strategy
├── 📄 train.py                 # Main training script (includes train_colab function)
├── 📄 dcrnn_model.py           # DCRNN model architecture
├── 📄 dcrnn_cell.py            # Graph convolution cell
├── 📄 dcrnn_supervisor.py      # Training supervisor (includes loss function)
├── 📄 utils.py                 # Data utilities and preprocessing
├── 📄 data_fetch.py            # Data fetching from Yahoo Finance
├── 📄 trading_strategy.py      # Trading strategy implementation
└── 📄 quick_demo.py            # Complete demo with trading strategy

### 📁 Output Files

After running the complete demo, you'll get:

- **`demo_trading_strategy.csv`** - Portfolio allocation and trading recommendations
- **`demo_predictions.csv`** - Multi-horizon price predictions for all stocks
- **`demo_logs/`** - Training logs, checkpoints, and TensorBoard files

## 🧠 Model Details

### DCRNN Architecture

- **Input Layer**: Multi-stock time series data
- **Encoder**: Processes historical sequences with graph convolution
- **Decoder**: Generates future predictions
- **Graph Convolution**: Captures inter-stock relationships
- **Attention Mechanism**: Focuses on relevant temporal patterns

### Hyperparameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `seq_len` | Input sequence length | 24 | 12-48 |
| `horizon` | Prediction horizon | 4 | 1-12 |
| `batch_size` | Training batch size | 32 | 16-128 |
| `rnn_units` | Hidden layer size | 64 | 32-256 |
| `num_rnn_layers` | Number of RNN layers | 2 | 1-4 |
| `max_diffusion_step` | Graph diffusion steps | 2 | 1-5 |

### Training Process

1. **Data Preparation**: Load and preprocess stock data
2. **Graph Construction**: Build adjacency matrix from correlations
3. **Model Initialization**: Create DCRNN with specified parameters
4. **Training Loop**: Iterative optimization with validation
5. **Checkpointing**: Save best models during training
6. **Evaluation**: Test on held-out data

## 📊 Data Requirements

### Input Format

The system expects CSV files with the following columns:

```csv
Date,Ticker,Open,High,Low,Close,Adj Close,Volume
2020-01-01,RELIANCE.BO,726.84,739.19,726.84,739.19,726.84,1452578
2020-01-01,TCS.BO,1922.33,1922.33,1911.39,1911.39,1922.33,56619
...
```

### Data Sources

- **Yahoo Finance**: Built-in support via `yfinance`
- **Custom Data**: Any CSV with required columns
- **Real-time Feeds**: Can be extended for live data

### Data Preprocessing

- **Normalization**: Z-score standardization
- **Windowing**: Overlapping time windows for sequences
- **Graph Construction**: Correlation-based adjacency matrices
- **Missing Data**: Forward/backward fill handling

## 🎯 Training & Inference

### Complete Pipeline Output

The enhanced framework now provides **end-to-end algorithmic trading** with automatic trading strategy generation:

1. **📊 Model Training**: DCRNN model training with validation
2. **🔮 Price Predictions**: Multi-horizon predictions for all stocks
3. **🎯 Trading Strategy**: Automated portfolio allocation and recommendations
4. **💾 Results Export**: CSV files for strategy and predictions

#### Example Trading Strategy Output

```
🎯 GENERATING TRADING STRATEGY
==================================================
📊 Predicted Returns for Next Period:
   INFY.BO: 0.0150 (1.50%)
   TCS.BO: 0.0120 (1.20%)
   RELIANCE.BO: 0.0080 (0.80%)

💼 PORTFOLIO ALLOCATION
   Capital: $100,000.00
   Stocks Selected: 3
   Equal Weight Allocation: 33.3% per stock

📈 TRADING RECOMMENDATIONS:
   INFY.BO: $33,333.33 (33.3%) - Expected: 1.50%
   TCS.BO: $33,333.33 (33.3%) - Expected: 1.20%
   RELIANCE.BO: $33,333.33 (33.3%) - Expected: 0.80%

💰 EXPECTED PORTFOLIO PERFORMANCE:
   Expected Return: 1.17%
   Expected Profit: $1,166.67
   Expected Portfolio Value: $101,166.67
```

### Training Configuration

```python
# Example training configuration
config = {
    'data': {
        'data_path': 'stock_data.csv',
        'seq_len': 24,
        'horizon': 4,
        'batch_size': 32,
        'val_ratio': 0.1,
        'test_ratio': 0.1
    },
    'model': {
        'rnn_units': 64,
        'num_rnn_layers': 2,
        'filter_type': 'random_walk'
    },
    'train': {
        'epochs': 50,
        'base_lr': 1e-3,
        'patience': 10
    }
}
```

### Monitoring Training

```bash
# View training progress
tensorboard --logdir logs

# Check model checkpoints
ls logs/checkpoints/
```

### Inference Pipeline

1. **Load Trained Model**: Restore from checkpoint
2. **Preprocess Input**: Apply same preprocessing as training
3. **Generate Predictions**: Forward pass through model
4. **Post-process Output**: Inverse transform predictions
5. **Generate Trading Signals**: Convert predictions to actions

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python3 train.py --batch_size 16
   ```

2. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Data Loading Issues**
   ```bash
   # Check data format
   head -5 your_data.csv
   # Ensure Date column is parseable
   ```

4. **Training Not Converging**
   ```bash
   # Reduce learning rate
   python3 train.py --base_lr 0.0001
   # Increase sequence length
   python3 train.py --seq_len 48
   ```

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed
- **Batch Size**: Adjust based on available memory
- **Data Loading**: Use SSD storage for faster I/O
- **Model Size**: Reduce `rnn_units` for faster training

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python3 -m pytest tests/

# Code formatting
black *.py

## 🚀 Enhanced Features (Latest Update)

### Complete Trading Strategy Pipeline

The framework now provides **end-to-end algorithmic trading** with automatic strategy generation:

#### 🎯 What You Get

1. **Automated Model Training**: DCRNN training with validation and checkpointing
2. **Intelligent Stock Selection**: Threshold-based filtering for risk management
3. **Portfolio Optimization**: Equal-weight allocation with performance metrics
4. **Trading Recommendations**: Specific buy amounts and expected returns
5. **Export Functionality**: CSV files for strategy and predictions

#### 📊 Strategy Customization

```python
# Conservative approach
strategy = generate_trading_strategy(
    capital=100000,
    threshold=0.02,    # 2% minimum return
    top_n=2            # Top 2 stocks only
)

# Aggressive approach  
strategy = generate_trading_strategy(
    capital=100000,
    threshold=0.005,   # 0.5% minimum return
    top_n=5            # Top 5 stocks
)
```

#### 🔄 Complete Workflow

```bash
# Run the complete pipeline
python3 quick_demo.py

# Output:
# ✅ Model training completed
# 🎯 Trading strategy generated  
# 💼 Portfolio allocation calculated
# 💾 Results saved to CSV files
```

#### 📈 Real-World Applications

- **Portfolio Management**: Automated rebalancing based on predictions
- **Risk Management**: Threshold-based stock filtering
- **Performance Tracking**: Expected returns and profit calculations
- **Strategy Backtesting**: Historical validation of trading approaches

The enhanced framework transforms your DCRNN predictions into **actionable trading strategies** with professional-grade portfolio management! 🎉

