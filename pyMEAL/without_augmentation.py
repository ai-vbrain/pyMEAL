__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '21st May, 2025'


import os
import tensorflow as tf
from . import basics as bcs
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Environment Configuration
bcs.configure_environment("1,3")
bcs.set_random_seed(42)

# Constants
BATCH_SIZE = 1
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]

# Model Components

def create_encoder(input_tensor):
    """
    Creates the encoder block with residual layers and downsampling. This encoder applies sequential 3D convolutions, residual blocks, 
    and max pooling to reduce spatial dimensions while extracting features.
    """
    x = layers.Conv3D(32, 3, activation='relu', padding='same')(input_tensor)
    x = bcs.refined_residual_block(x, 32)
    x = layers.MaxPooling3D(2)(x)

    x = layers.Conv3D(64, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 64)
    x = layers.MaxPooling3D(2)(x)

    x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
    return layers.Dropout(0.4)(x)


def refined_diffusion_model(input_shape):
    """
    Builds the refined diffusion model architecture. This model consists of a 3D encoder-decoder structure with residual blocks,
    downsampling via max pooling, and upsampling via transposed convolutions.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    features = create_encoder(inputs)

    # Decoder
    x = layers.UpSampling3D(2)(features)
    x = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 128)

    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3DTranspose(64, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 64)

    outputs = layers.Conv3D(1, 3, activation='sigmoid', padding='same')(x)

    return Model(inputs=inputs, outputs=outputs)

##
