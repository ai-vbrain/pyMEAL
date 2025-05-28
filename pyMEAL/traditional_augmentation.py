__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '21st May, 2025'

import os
import tensorflow as tf
from . import basics as bcs
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
# Environment Configuration
bcs.configure_environment("1,2")
bcs.set_random_seed(42)

# Constants
BATCH_SIZE = 1
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]  # Spatial dimensions only

# --- Augmentation Layers ---
class FlipAugmentation(layers.Layer):
    def call(self, inputs):
        inputs = tf.cond(tf.random.uniform([]) > 0.5, lambda: tf.reverse(inputs, [1]), lambda: inputs)
        inputs = tf.cond(tf.random.uniform([]) > 0.5, lambda: tf.reverse(inputs, [2]), lambda: inputs)
        return inputs


class RotateAugmentation(tf.keras.layers.Layer):
    """3D rotation augmentation along specified axes."""

    def __init__(self, axis=(1, 2), **kwargs):
        super().__init__(**kwargs)
        self._axis = axis  # store as a private member to avoid base layer conflict

    def call(self, inputs):
        if self._axis == (1, 2):  # Rotate along height and width
            rotated = tf.transpose(inputs, perm=[0, 2, 1, 3, 4])
            rotated = tf.reverse(rotated, axis=[2])
        elif self._axis == (1, 3):  # Rotate along depth and height
            rotated = tf.transpose(inputs, perm=[0, 3, 2, 1, 4])
            rotated = tf.reverse(rotated, axis=[2])
        elif self._axis == (2, 3):  # Rotate along depth and width
            rotated = tf.transpose(inputs, perm=[0, 1, 3, 2, 4])
            rotated = tf.reverse(rotated, axis=[3])
        else:
            raise ValueError("Invalid axis for rotation.")
        return rotated

class CropAugmentation(tf.keras.layers.Layer):
    def __init__(self, crop_proportion=0.8, target_shape=(128, 128, 64, 1), **kwargs):
        super().__init__(**kwargs)
        self.crop_proportion = crop_proportion
        self.target_shape = target_shape

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, h, w, d, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_shape[4]

        crop_h = tf.cast(tf.cast(h, tf.float32) * self.crop_proportion, tf.int32)
        crop_w = tf.cast(tf.cast(w, tf.float32) * self.crop_proportion, tf.int32)
        crop_d = tf.cast(tf.cast(d, tf.float32) * self.crop_proportion, tf.int32)

        offset_h = tf.random.uniform([], 0, h - crop_h + 1, dtype=tf.int32)
        offset_w = tf.random.uniform([], 0, w - crop_w + 1, dtype=tf.int32)
        offset_d = tf.random.uniform([], 0, d - crop_d + 1, dtype=tf.int32)

        cropped = inputs[:, offset_h:offset_h + crop_h,
                         offset_w:offset_w + crop_w,
                         offset_d:offset_d + crop_d, :]

        target_h, target_w, target_d, _ = self.target_shape

        def resize_volume(volume):
            # Resize spatial dimensions
            volume = tf.transpose(volume, [2, 0, 1, 3])  # (D, H, W, C)
            volume = tf.map_fn(lambda x: tf.image.resize(x, [target_h, target_w]), volume)
            volume = tf.transpose(volume, [1, 2, 0, 3])  # (H, W, D, C)

            current_d = tf.shape(volume)[2]
            depth_diff = target_d - current_d

            def pad():
                return tf.pad(volume, [[0, 0], [0, 0], [0, depth_diff], [0, 0]])

            def crop():
                return volume[:, :, :target_d, :]

            volume = tf.cond(depth_diff > 0, pad, crop)
            return volume

        resized_batch = tf.map_fn(resize_volume, cropped, fn_output_signature=tf.float32)
        resized_batch.set_shape((None,) + self.target_shape)

        return resized_batch


class IntensityAugmentation(layers.Layer):
    def call(self, inputs):
        x = tf.image.random_brightness(inputs, max_delta=0.1)
        x = tf.image.random_contrast(x, 0.9, 1.1)
        return x


def create_dataset(input_dir, target_dir, batch_size=1, shuffle=True):
    """
    Creates a TensorFlow dataset with 3D augmentation and batching.
    """
    base_ds = bcs.create_dataset(
        input_dir, target_dir,
        input_shape=(128, 128, 64, 1),
        target_shape=(128, 128, 64, 1),
        batch_size=batch_size,
        shuffle=shuffle
    )

    def apply_augmentations(input_vol, target_vol):
        flip = FlipAugmentation()
        rotate = RotateAugmentation(axis=(1, 2))
        crop = CropAugmentation(crop_proportion=0.8, target_shape=(128, 128, 64, 1))
        intensity = IntensityAugmentation()

        augmentations = [flip, rotate, crop, intensity]
        for aug in augmentations:
            input_vol = aug(input_vol)
            target_vol = aug(target_vol)
        return input_vol, target_vol

    return base_ds.map(apply_augmentations, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# --- Model Architecture ---
def refined_diffusion_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(64, 3, activation='relu', padding='same')(inputs)
    x = bcs.refined_residual_block(x, 64)
    x = layers.MaxPooling3D(2)(x)

    x = layers.Conv3D(128, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 128)
    x = layers.MaxPooling3D(2)(x)

    x = layers.Conv3D(256, 3, activation='relu', padding='same')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3DTranspose(128, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 128)

    x = layers.UpSampling3D(2)(x)
    x = layers.Conv3DTranspose(64, 3, activation='relu', padding='same')(x)
    x = bcs.refined_residual_block(x, 64)

    outputs = layers.Conv3D(1, 3, activation='sigmoid', padding='same')(x)
    return keras.Model(inputs, outputs, name='refined_diffusion_model')

