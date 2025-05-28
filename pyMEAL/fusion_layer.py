__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '21st May, 2025'


import os
import tensorflow as tf
from . import basics as bcs
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List

# Environment Configuration
bcs.configure_environment("2,1")
bcs.set_random_seed(42)

# Constants
BATCH_SIZE = 1
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]  


def create_dataset(input_dir: str,
                   target_dir: str,
                   batch_size: int = BATCH_SIZE,
                   shuffle: bool = True) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset pipeline with multi-stream augmentation keys.
    This function uses a base preprocessing pipeline and maps each input volume
    into named streams to be handled by different augmentation branches in the model.

    Inputs
    ----------
    - input_dir : str, directory containing the input NIfTI files.
    - target_dir : str, directory containing the target NIfTI files.
    - batch_size : int (optional),number of samples per batch (default is BATCH_SIZE).
    - shuffle : bool (optional), whether to shuffle the dataset (default is True).

    Returns
    -------
    tf.data.Dataset, a TensorFlow dataset that yields a dictionary of input streams and targets.
    """
    base_ds = bcs.create_dataset(
        input_dir,
        target_dir,
        INPUT_SHAPE,
        TARGET_SHAPE,
        batch_size,
        shuffle
    )

    def create_augmentation_streams(input_vol, target_vol):
        """
        Duplicate input volume for different augmentation streams.

        Inputs
        ----------
        - input_vol : tf.Tensor, the preprocessed input volume.
        - target_vol : tf.Tensor, the corresponding target volume.

        Returns
        -------
        - dict, tf.Tensor, a dictionary with augmentation stream names and the original input volume,
        along with the target volume.
        """
        return {
            'flip_stream': input_vol,
            'rotate_stream': input_vol,
            'crop_stream': input_vol,
            'intensity_stream': input_vol
        }, target_vol

    dataset = base_ds.map(create_augmentation_streams, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


class FeatureFusionLayer(keras.layers.Layer):
    """
    A custom layer for fusing multiple 3D feature maps using 1x1x1 convolution.
    This layer concatenates a list of input feature maps along the channel dimension
    and applies a 3D 1x1x1 convolution to produce a fused output.

    Parameters
    ----------
    units : int
        Number of output filters for the fusion convolution.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.conv = layers.Conv3D(filters=units, kernel_size=1, padding='same')

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        """
        Forward pass for the feature fusion layer.

        Parameters
        ----------
        inputs : List[tf.Tensor]
            A list of 5D tensors to be fused (shape: [batch, height, width, depth, channels]).

        Returns
        -------
        tf.Tensor
            A fused tensor with shape [batch, height, width, depth, units].
        """
        concatenated_inputs = tf.concat(inputs, axis=-1)
        fused_output = self.conv(concatenated_inputs)
        return fused_output

    def get_config(self) -> dict:
        """
        Return the configuration of the layer for serialization.

        Returns
        -------
        dict
            A dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({'units': self.units})

        return config


# Model Architecture
def build_multistream_model(input_shape):
    """Build multi-stream model with different augmentation pathways."""
    # Input streams
    inputs = {
        'flip_stream': layers.Input(input_shape, name='flip_stream'),
        'rotate_stream': layers.Input(input_shape, name='rotate_stream'),
        'crop_stream': layers.Input(input_shape, name='crop_stream'),
        'intensity_stream': layers.Input(input_shape, name='intensity_stream')
    }

    # Shared encoder architecture
    def encoder(x):
        """Encoder with residual blocks and downsampling."""
        x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
        x = bcs.refined_residual_block(x, 64)
        x = layers.MaxPooling3D(2)(x)
        
        x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
        x = bcs.refined_residual_block(x, 128)
        x = layers.MaxPooling3D(2)(x)
        
        x = layers.Conv3D(256, 3, activation='relu', padding='same')(x)
        return layers.Dropout(0.4)(x)

    # Process each stream through encoder
    features = [encoder(inputs[name]) for name in inputs]
    
    # Feature fusion
    fused_features = FeatureFusionLayer(128)(features)

    # Decoder
    x = layers.UpSampling3D(2)(fused_features)
    x = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 128)
    
    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3DTranspose(64, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 64)

    # Output
    outputs = layers.Conv3D(1, 3, activation='sigmoid', padding='same')(x)

    return keras.Model(inputs=list(inputs.values()), outputs=outputs)









