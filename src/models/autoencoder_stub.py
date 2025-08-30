# src/models/autoencoder_stub.py
# ----------------------------------------------------------------------
# Autoencoder Stub for Fraud Detection (Anomaly Detection)
# ----------------------------------------------------------------------
# This module provides a ready-to-use Keras-based autoencoder implementation
# tailored for fraud detection tasks. It includes:
#   1. build_autoencoder(): builds a symmetric dense autoencoder.
#   2. calculate_reconstruction_error(): computes reconstruction errors.
#
# Why Autoencoders for Anomaly Detection?
# ---------------------------------------
# Autoencoders learn to reconstruct input data by compressing (encoding)
# and then decompressing (decoding) it. When trained only on "normal"
# transactions, they excel at reconstructing such data. However, for
# anomalous or fraudulent transactions (patterns not seen during training),
# the reconstruction error will typically be higher. Thus, reconstruction
# error becomes a useful anomaly score.
#
# Thresholds:
# -----------
# After training, one common method is to analyze reconstruction errors on
# validation data. A threshold can be set at a high quantile (e.g., 95th or
# 99th percentile). Samples exceeding this error are flagged as anomalies.
#
# NOTE: This file is a stub scaffold for future development and experimentation.
# It does not auto-train or auto-run — see the commented example for usage.
# ----------------------------------------------------------------------

import numpy as np  # Added numpy import
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense

def build_autoencoder(input_dim, latent_dim=2, hidden_layers=[64, 32, 16]):
    """
    Build a symmetric autoencoder using the Keras Functional API.

    Parameters
    ----------
    input_dim : int
        Number of features in the input data.
    latent_dim : int, optional (default=2)
        Dimension of the bottleneck (compressed latent representation).
    hidden_layers : list of int, optional
        List specifying the units for hidden encoder layers (decreasing).
        The decoder will mirror these layers in reverse order.

    Returns
    -------
    autoencoder : keras.Model
        Full autoencoder model (input -> reconstruction).
    encoder : keras.Model
        Encoder model (input -> latent representation).
    """
    # Input layer
    inputs = Input(shape=(input_dim,), name="input")

    # Encoder part
    x = inputs
    for i, units in enumerate(hidden_layers):
        x = Dense(units, activation="relu", name=f"encoder_dense_{i}")(x)

    # Bottleneck / Latent representation
    latent = Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder part (mirror of encoder)
    x = latent
    for i, units in enumerate(reversed(hidden_layers)):
        x = Dense(units, activation="relu", name=f"decoder_dense_{i}")(x)

    # Output layer (sigmoid, assuming inputs are normalized to [0,1])
    outputs = Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    # Models
    autoencoder = Model(inputs, outputs, name="autoencoder")
    encoder = Model(inputs, latent, name="encoder")

    return autoencoder, encoder


def calculate_reconstruction_error(autoencoder, X_data):
    """
    Compute the reconstruction error for each sample.

    Parameters
    ----------
    autoencoder : keras.Model
        Trained autoencoder model.
    X_data : np.ndarray
        Input data (samples x features).

    Returns
    -------
    errors : np.ndarray
        Mean squared reconstruction error per sample.
    """
    # Get reconstructed data
    reconstructions = autoencoder.predict(X_data, verbose=0)

    # Compute mean squared error per sample
    errors = np.mean(np.square(X_data - reconstructions), axis=1)
    return errors


# ----------------------------------------------------------------------
# Example Usage (commented out for educational purposes)
# ----------------------------------------------------------------------
"""
Example Workflow for Fraud Detection with Autoencoder
=====================================================

1. Data Preparation
-------------------
from sklearn.model_selection import train_test_split

# Assume `X` is your transaction feature matrix (numpy array).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

2. Build Model
--------------
from tensorflow.keras.callbacks import EarlyStopping

input_dim = X_train.shape[1]
autoencoder, encoder = build_autoencoder(input_dim, latent_dim=8, hidden_layers=[64, 32, 16])

autoencoder.compile(optimizer="adam", loss="mse")

3. Train Model
--------------
# Use early stopping to prevent overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

4. Evaluate with Reconstruction Error
-------------------------------------
errors_train = calculate_reconstruction_error(autoencoder, X_train)
errors_test = calculate_reconstruction_error(autoencoder, X_test)

# Set threshold (e.g., 95th percentile of train error)
threshold = np.percentile(errors_train, 95)

# Flag anomalies
anomalies = errors_test > threshold

5. Save Model and Scaler
------------------------
import joblib

autoencoder.save("models/autoencoder_model")
joblib.dump(scaler, "models/scaler.pkl")

6. Interpret Results
--------------------
- Low reconstruction error = sample is likely normal.
- High reconstruction error = sample may be anomalous.
- The threshold is tunable depending on false positive/negative trade-offs.
"""


# ----------------------------------------------------------------------
# Main Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple demo of building the model and showing summary
    dummy_input_dim = 20  # pretend we have 20 features
    autoencoder, encoder = build_autoencoder(dummy_input_dim, latent_dim=4)

    print("Autoencoder Summary:")
    autoencoder.summary()

    print("\nEncoder Summary:")
    encoder.summary()