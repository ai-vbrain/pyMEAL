__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '20th May, 2025'



import os
import tensorflow as tf
from . import basics as bcs
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from . import builder_block as bd

# Constants and hyperparameters
BATCH_SIZE = 4
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]  # Spatial dimensions only

# # File paths

def create_dataset(input_dir, target_dir, batch_size=BATCH_SIZE, shuffle=True):
    """
    Creates a TensorFlow dataset pipeline with duplicated augmented input streams.
    This function wraps a base dataset loader and duplicates each input volume
    into four parallel streams (e.g., for multi-branch model inputs), without applying
    actual augmentations. The target remains unchanged.

    Inputs:
    ----------
    input_dir : str, directory containing input NIfTI files.
    target_dir : str, directory containing target NIfTI files.
    batch_size : int (optional), number of samples per batch. Default is `BATCH_SIZE`.
    shuffle : bool (optional), whether to shuffle the dataset. Default is True.

    Return:
    -------

    tf.data.Dataset, a dataset yielding tuples of ((input1, input2, input3, input4), target) per batch.
    """
    base_ds = bcs.create_dataset(
        input_dir, target_dir,
        INPUT_SHAPE, TARGET_SHAPE,
        batch_size, shuffle
    )

    def duplicate_inputs(input_vol, target_vol):
        """Duplicates each input volume four times for multi-stream input."""
        return (input_vol, input_vol, input_vol, input_vol), target_vol

    dataset = base_ds.map(duplicate_inputs, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_multistream_model(input_shape):
    """
    Builds a multi-stream 3D CNN model with shared encoders and feature fusion.
    Each input stream passes through a shared encoder architecture to extract features.
    The encoded features from all four streams (flip, rotate, crop, intensity) are
    concatenated and decoded through upsampling layers to generate the final output.

    Inputs:
    ----------
    input_shape : tuple, shape of each individual input volume (e.g., (64, 64, 32, 1)).

    Return:
    -------
    keras.Model, a compiled Keras model that accepts four input streams and outputs a single reconstructed 3D volume.
    """
    # Define input streams
    flip_stream = keras.Input(shape=input_shape, name='flip_stream')
    rotate_stream = keras.Input(shape=input_shape, name='rotate_stream')
    crop_stream = keras.Input(shape=input_shape, name='crop_stream')
    intensity_stream = keras.Input(shape=input_shape, name='intensity_stream')

    # Shared encoder architecture
    def encoder(input_tensor):
        """Encoder with residual blocks and downsampling."""
        x = layers.Conv3D(64, 3, activation='relu', padding='same')(input_tensor)
        x = bcs.refined_residual_block(x, 64)
        x = layers.MaxPooling3D(2)(x)

        x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
        x = bcs.refined_residual_block(x, 128)
        x = layers.MaxPooling3D(2)(x)

        x = layers.Conv3D(256, 3, activation='relu', padding='same')(x)
        x = layers.Dropout(0.4)(x)
        return x

    # Process each stream through encoder
    features = [
        encoder(flip_stream),
        encoder(rotate_stream),
        encoder(crop_stream),
        encoder(intensity_stream)
    ]

    # Feature fusion
    fused_features = layers.Concatenate()(features)

    # Decoder with upsampling
    x = layers.UpSampling3D(2)(fused_features)
    x = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 128)

    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3DTranspose(64, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 64)

    # Output layer
    outputs = layers.Conv3D(1, 3, activation='sigmoid', padding='same')(x)

    return keras.Model(
        inputs=[flip_stream, rotate_stream, crop_stream, intensity_stream],
        outputs=outputs
    )