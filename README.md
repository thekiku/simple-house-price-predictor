# California Housing Prediction (TensorFlow/Keras)

A basic implementation of a regression pipeline using TensorFlow and Keras to predict house prices using the classic California Housing dataset.

## Core Setup
* **Data Processing:** Scaled features via `StandardScaler` and split into clean Train/Validation/Test partitions.
* **Model Architecture:** Fully connected neural network built using the Keras Sequential API.
* **Regularization:** Layer stabilization using Batch Normalization, Dropout, and L2 regularization to manage model variance.
* **Training Dynamics:** Regulated training flow via `EarlyStopping` and dynamic learning rate adjustments (`ReduceLROnPlateau`).
<img width="678" height="470" alt="image" src="https://github.com/user-attachments/assets/80937765-5452-4e7e-8418-7d22e2db70e9" />


## Quick Start

1. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn numpy matplotlib
