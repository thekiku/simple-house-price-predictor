# California Housing Prediction (TensorFlow/Keras)

A clean, minimalist implementation of a regression pipeline using TensorFlow 2.x and Keras to predict house prices using the classic California Housing dataset.

## Core Setup
* **Data Processing:** Scaled features via `StandardScaler` and split into clean Train/Validation/Test partitions.
* **Model Architecture:** Fully connected neural network built using the Keras Sequential API.
* **Regularization:** Layer stabilization using Batch Normalization, Dropout, and L2 regularization to manage model variance.
* **Training Dynamics:** Regulated training flow via `EarlyStopping` and dynamic learning rate adjustments (`ReduceLROnPlateau`).

## Quick Start

1. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn numpy matplotlib
