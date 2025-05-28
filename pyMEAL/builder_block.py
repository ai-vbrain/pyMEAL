__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '20th May, 2025'


import os
import tensorflow as tf
from . import basics as bcs
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from tensorflow import keras
from typing import Tuple

# Environment configuration
bcs.configure_environment("2,1")
bcs.set_random_seed(42)

# Constants
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 1



class FlipAugmentation(layers.Layer):
    """
    A custom Keras layer that randomly flips input volumes along height and width axes.

    This layer performs random up-down and left-right flipping using `tf.cond` and 
    `tf.reverse`, helping to increase spatial variability during training.

    Returns
    -------
    tf.Tensor
        The augmented tensor with the same shape as the input.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Generate two random values to decide flip axes
        flip_coin = tf.random.uniform(shape=[2], minval=0, maxval=1)

        # Apply vertical flip with 50% probability
        inputs = tf.cond(
            flip_coin[0] > 0.5,
            lambda: tf.reverse(inputs, axis=[1]),  # Flip vertically
            lambda: inputs
        )

        # Apply horizontal flip with 50% probability
        inputs = tf.cond(
            flip_coin[1] > 0.5,
            lambda: tf.reverse(inputs, axis=[2]),  # Flip horizontally
            lambda: inputs
        )

        return inputs


class RotateAugmentation(layers.Layer):
    """
    A custom Keras layer that randomly rotates 3D volumes by 90-degree increments.

    This augmentation performs in-plane rotation (around the axial plane) by 
    reshaping the 5D tensor into 2D slices, applying `tf.image.rot90`, and then 
    restoring the original 5D shape.

    Return
    -------
    tf.Tensor, rotated tensor with the same shape as the input.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Random integer in [0, 3] representing number of 90-degree rotations
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        original_shape = tf.shape(inputs)

        # Reshape: merge batch and depth for 2D rotation
        reshaped = tf.reshape(
            inputs,
            [-1, original_shape[1], original_shape[2], original_shape[4]]
        )

        # Apply 2D 90-degree rotation k times
        rotated = tf.image.rot90(reshaped, k=k)

        # Reshape back to original 5D shape
        rotated = tf.reshape(
            rotated,
            tf.stack([
                original_shape[0],
                original_shape[1],
                original_shape[2],
                original_shape[3],
                original_shape[4]
            ])
        )

        return rotated


class CropAugmentation(keras.layers.Layer):
    """
    Randomly crops and resizes input volumes for augmentation.

    This layer performs a center crop on 5D input tensors and resizes the result
    back to the original spatial dimensions (height, width). It is useful for 
    enforcing spatial invariance during training of 3D medical image models.

    Inputs
    ----------
    - crop_size : Tuple[int], the shape of the cropped region (height, width, depth, channels).
    - original_size : Tuple[int], the shape to which cropped volumes are resized (height, width, depth, channels).
    """

    def __init__(self, crop_size: Tuple[int], original_size: Tuple[int], **kwargs):
        super(CropAugmentation, self).__init__(**kwargs)
        self.crop_size = crop_size
        self.original_size = original_size

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        offset_height = (self.original_size[0] - self.crop_size[0]) // 2
        offset_width = (self.original_size[1] - self.crop_size[1]) // 2
        offset_depth = (self.original_size[2] - self.crop_size[2]) // 2

        cropped = inputs[
            :,
            offset_height:offset_height + self.crop_size[0],
            offset_width:offset_width + self.crop_size[1],
            offset_depth:offset_depth + self.crop_size[2],
            :
        ]

        batch_size = tf.shape(cropped)[0]
        cropped_reshaped = tf.reshape(
            cropped,
            [batch_size, self.crop_size[0], self.crop_size[1],
             self.crop_size[2] * self.crop_size[3]]
        )

        resized = tf.image.resize(
            cropped_reshaped,
            size=(self.original_size[0], self.original_size[1])
        )

        resized = tf.reshape(
            resized,
            [batch_size, self.original_size[0], self.original_size[1],
             self.crop_size[2], self.crop_size[3]]
        )

        return resized
    

class IntensityAugmentation(layers.Layer):
    """
    A custom Keras layer to apply random brightness and contrast adjustments.
    This layer introduces variability in intensity to improve model robustness to
    lighting conditions and scanner variability in medical imaging.

    Return
    -------
    tf.Tensor, a tensor with the same shape as the input, with randomized intensity.
    """

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        augmented = tf.image.random_brightness(inputs, max_delta=0.1)
        augmented = tf.image.random_contrast(augmented, lower=0.9, upper=1.1)
        return augmented


def create_dataset(input_dir: str, target_dir: str) -> tf.data.Dataset:
    """
    Create a TensorFlow dataset pipeline with preprocessing and augmentation.
    This function wraps the dataset creation using predefined constants for input/output
    shapes and batch size. It relies on the `bcs.create_dataset()` utility.

    Inputs
    ----------
    - input_dir : str, directory containing input NIfTI files.
    - target_dir : str,directory containing target NIfTI files.

    Return
    -------
    tf.data.Dataset, a batched and prefetched TensorFlow dataset.
    """
    base_ds = bcs.create_dataset(
        input_dir,
        target_dir,
        INPUT_SHAPE,
        TARGET_SHAPE,
        BATCH_SIZE
    )
    return base_ds


def refined_diffusion_model(input_shape: Tuple[int] = INPUT_SHAPE) -> Model:
    """
    Create a refined 3D encoder model for diffusion-based architectures.
    This model extracts multi-scale features using a combination of 
    convolutional blocks, residual connections, and pooling. The 
    encoder is suitable for 3D image-to-image tasks like medical 
    image translation or reconstruction.

    Inputs
    ----------
    input_shape : Tuple[int] (optional) shape of the input 3D volume (default is INPUT_SHAPE).

    Return
    -------
    tf.keras.Model, Keras model representing the feature encoder portion of a diffusion model.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder block 1
    conv1 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = bcs.refined_residual_block(conv1, 64)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    # Encoder block 2
    conv2 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = bcs.refined_residual_block(conv2, 128)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Dropout(0.4)(conv3)

    return Model(inputs, conv3, name="refined_diffusion_model")


def build_controller_block(features: tf.Tensor) -> Model:
    """
    Builds a controller block to dynamically weight feature inputs.
    This small network takes a 2D tensor (e.g., from a time series or feature map),
    applies two dense layers, and generates learned weights to re-weight the input features.

    Inputs
    ----------
    features : tf.Tensor, a sample tensor whose shape determines the input shape of the controller block,
        typically (batch_size, timesteps, features).

    Input
    -------
    tf.keras.Model, a Keras model that performs weighted modulation of input features.
    """
    input_shape = (features.shape[1], features.shape[2])
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(input_shape[1], activation='sigmoid')(x)  # Use sigmoid for scalar weights

    outputs = layers.Multiply()([inputs, x])
    return Model(inputs=inputs, outputs=outputs, name="controller_block")


def build_multistream_model(input_shape: Tuple[int] = INPUT_SHAPE) -> Model:
    """
    Builds a multi-stream augmentation-aware image translation model.
    This architecture applies four fixed augmentations (flip, rotate, crop, intensity)
    to the input, extracts features from each augmented version using a shared encoder,
    weights the features with a learned controller, and reconstructs the image through a decoder.

    Inputs
    ----------
    input_shape : Tuple[int], optional
        The shape of the input 3D volume (default is INPUT_SHAPE).

    Return
    -------
    tf.keras.Model, a compiled Keras model that performs generalizable image translation
    with augmentation-informed feature fusion.
    """
    inputs = layers.Input(shape=input_shape)

    # Augmentation streams
    flip_stream = FlipAugmentation()(inputs)
    rotate_stream = RotateAugmentation()(inputs)
    crop_stream = CropAugmentation(
        crop_size=(64, 64, input_shape[2], 1),
        original_size=input_shape
    )(inputs)
    intensity_stream = IntensityAugmentation()(inputs)

    # Shared encoder
    encoder = refined_diffusion_model(input_shape)
    flip_features = encoder(flip_stream)
    rotate_features = encoder(rotate_stream)
    crop_features = encoder(crop_stream)
    intensity_features = encoder(intensity_stream)

    # Global pooling to reduce spatial dimensions
    flip_output = layers.GlobalAveragePooling3D()(flip_features)
    rotate_output = layers.GlobalAveragePooling3D()(rotate_features)
    crop_output = layers.GlobalAveragePooling3D()(crop_features)
    intensity_output = layers.GlobalAveragePooling3D()(intensity_features)

    # Stack all feature streams into a single tensor (batch, 4, features)
    all_features = layers.Lambda(lambda x: tf.stack(x, axis=1))([
        flip_output, rotate_output, crop_output, intensity_output
    ])

    # Controller block learns dynamic weights per feature stream
    controller = build_controller_block(all_features)
    weighted_features = controller(all_features)

    # Sum across streams to get fused representation
    combined_features = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_features)

    # Decode fused features back into image space
    x = layers.Dense(32 * 32 * 16 * 64, activation='relu')(combined_features)
    x = layers.Reshape((32, 32, 16, 64))(x)

    x = layers.UpSampling3D((2))(x)
    x = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 128)

    x = layers.UpSampling3D((2))(x)
    x = layers.Conv3DTranspose(64, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 64)

    outputs = layers.Conv3D(1, 3, activation='sigmoid', padding='same', name='reconstructed_output')(x)

    return Model(inputs=inputs, outputs=outputs, name="multistream_augmentation_model")



