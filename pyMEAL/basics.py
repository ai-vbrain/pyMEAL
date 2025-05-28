__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '20th May, 2025'


import os
import tensorflow as tf
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorflow import keras
from tensorflow.keras import layers
import math
import tensorflow_addons as tfa


# Environment Configuration
def configure_environment(gpu_ids="1,3"):
    """Configure TensorFlow environment and GPU settings."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def set_random_seed(seed_value=42):
    """Set random seeds for reproducibility."""
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)


def apply_windowing(ct_data, window_level=40, window_width=80):
    """
    Apply windowing to CT data to enhance visualization of specific tissue ranges.

    Inputs
    ----------
    - ct_data : np.ndarray, the input CT image data as a NumPy array.
    - window_level : float (optional), the center of the intensity window (default is 40).
    - window_width : float (optional), the width of the intensity window (default is 80).

    Return
    -------
    np.ndarray: Windowed and normalized CT image with values scaled between 0 and 1.
    """
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    windowed = np.clip(ct_data, lower, upper)
    return (windowed - lower) / (upper - lower)



def resize_volume(volume, target_shape):
    """
    Resize a 3D volume to a target shape using trilinear interpolation.

    Inputs
    ----------
    - volume : np.ndarray, the input 3D volume to be resized.
    - target_shape : tuple of int, the desired output shape as (depth, height, width).

    Returns
    -------
    np.ndarray, the resized 3D volume.
    """
    scale_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, scale_factors, order=1)


def normalize_volume(volume):
    """
    Normalize a 3D volume to the [0, 1] range.

    Input
    ----------
    - volume : np.ndarray, Input 3D volume.

    Return
    -------
    np.ndarray, Normalized volume with values scaled between 0 and 1.
    """
    return (volume - np.min(volume)) / (np.ptp(volume) + 1e-8)


def load_and_preprocess(file_pair, target_shape):
    """
    Load and preprocess a pair of NIfTI files (input and target volumes).

    This function applies CT windowing to the input volume, resizes both volumes
    to the target shape using trilinear interpolation, normalizes them to [0, 1],
    and adds a channel dimension.

    Inputs
    ----------
    - file_pair : tuple of tf.Tensor, tuple containing file paths to the input and target NIfTI volumes.
    - target_shape : tuple of int, desired shape for the volumes as (depth, height, width).

    Return
    -------
    tuple of np.ndarray, preprocessed input and target volumes with shape (D, H, W, 1), dtype float32.
    """
    input_path, target_path = file_pair

    # Load and decode NIfTI files
    input_vol = nib.load(input_path.numpy().decode('utf-8')).get_fdata()
    target_vol = nib.load(target_path.numpy().decode('utf-8')).get_fdata()

    # Apply CT windowing to input
    input_vol = apply_windowing(input_vol)

    # Resize and normalize
    input_vol = normalize_volume(resize_volume(input_vol, target_shape))
    target_vol = normalize_volume(resize_volume(target_vol, target_shape))

    # Add channel dimension
    input_vol = np.expand_dims(input_vol, axis=-1).astype(np.float32)
    target_vol = np.expand_dims(target_vol, axis=-1).astype(np.float32)

    return input_vol, target_vol


def tf_load_and_preprocess(input_path, target_path, input_shape, target_shape):
    """
    TensorFlow-compatible wrapper for loading and preprocessing NIfTI volumes.

    This function wraps the NumPy-based `load_and_preprocess` using `tf.py_function`
    so it can be used in a `tf.data.Dataset` pipeline. It ensures that the resulting
    tensors have defined shapes for downstream model compatibility.

    Inputs
    ----------
    - input_path : tf.Tensor, path to the input NIfTI volume (as a string tensor).
    - target_path : tf.Tensor, path to the target NIfTI volume (as a string tensor).
    - input_shape : tuple, desired shape of the input tensor (e.g., (D, H, W, C)).
    - target_shape : tuple, target shape for resizing before returning volumes.

    Return
    -------
    tuple of tf.Tensor, preprocessed input and target volumes with shape `input_shape`.
    """
    input_vol, target_vol = tf.py_function(
        func=lambda x: load_and_preprocess(x, target_shape),
        inp=[(input_path, target_path)],
        Tout=[tf.float32, tf.float32]
    )

    input_vol.set_shape(input_shape)
    target_vol.set_shape(input_shape)

    return input_vol, target_vol


def create_dataset(input_dir, target_dir, input_shape, target_shape, 
                   batch_size=1, shuffle=True):
    """
    Create a TensorFlow data pipeline for paired NIfTI volumes.

    This function builds a `tf.data.Dataset` that loads, preprocesses, batches,
    and optionally shuffles paired input and target volumes stored in `.nii.gz` format.

    Inputs
    ----------
    - input_dir : str, directory containing input NIfTI files (e.g., CT volumes).
    - target_dir : str, directory containing target NIfTI files (e.g., MRI volumes).
    - input_shape : tuple of int, final shape of the input volume tensors (after resizing and adding channel dimension).
    - target_shape : tuple of int, desired spatial shape to resize both input and target volumes to.
    - batch_size : int (optional), number of samples per batch (default is 1).
    - shuffle : bool (optional), whether to shuffle the dataset (default is True).

    Return
    -------
    tf.data.Dataset, a TensorFlow dataset yielding (input_volume, target_volume) batches.
    """
    input_files = sorted([
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('.nii.gz')
    ])
    
    target_files = sorted([
        os.path.join(target_dir, f) 
        for f in os.listdir(target_dir) 
        if f.endswith('.nii.gz')
    ])

    dataset = tf.data.Dataset.from_tensor_slices((input_files, target_files))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input_files), reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda x, y: tf_load_and_preprocess(x, y, input_shape, target_shape),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def refined_residual_block(inputs, filters, dropout_rate=0.2):
    """
    Apply a 3D residual block with two convolutional layers and dropout.

    This block enhances feature learning while maintaining skip connections
    and includes dropout regularization after the first convolution.

    Inputs
    ----------
    - inputs : tf.Tensor, input tensor of shape (D, H, W, C).
    - filters : int, number of filters for the convolutional layers.
    - dropout_rate : float (optional), dropout rate applied after the first convolution (default is 0.2).

    Return
    -------
    tf.Tensor, Output tensor after applying the residual block.
    """
    x = layers.Conv3D(filters, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.Dropout(rate=dropout_rate)(x)
    x = layers.Conv3D(filters, kernel_size=3, activation='relu', padding='same')(x)
    return layers.Add()([inputs, x])


def ssim_loss(y_true, y_pred):
    """
    Compute the Structural Similarity Index (SSIM) loss.

    SSIM loss is defined as 1 - mean SSIM, encouraging perceptual similarity 
    between the predicted and ground truth images.

    Inputs
    ----------
    - y_true : tf.Tensor, ground truth image tensor.
    - y_pred : tf.Tensor, predicted image tensor.

    Return
    -------
    tf.Tensor, scalar loss value representing (1 - mean SSIM).
    """
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def combined_loss(y_true, y_pred):
    """
    Compute a combined loss using L1 and SSIM components.

    This loss function balances pixel-wise accuracy (L1 loss) with perceptual 
    similarity (SSIM loss), helping to improve both structure and detail 
    preservation in image predictions.

    Inputs
    ----------
    - y_true : tf.Tensor, ground truth image tensor.
    - y_pred : tf.Tensor, predicted image tensor.

    Input
    -------
    tf.Tensor, scalar loss value combining L1 and SSIM losses.
    """
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return l1_loss + 0.8 * ssim_loss(y_true, y_pred)


def evaluate_model(model, dataset, input_shape):
    """
    Evaluate a model using PSNR and SSIM metrics on a validation dataset.

    This function iterates over a dataset of input-target pairs, makes predictions,
    extracts the middle slice for each 3D volume, and computes PSNR and SSIM.
    It also visualizes the first predicted sample.

    Inputs
    ----------
    - model : tf.keras.Model, the trained model to evaluate.
    - dataset : tf.data.Dataset, dataset yielding dictionaries with 'input' and 'target' keys.
    - input_shape : tuple of int, expected input shape (used to determine the middle slice).

    Returns
    -------
    - mean_psnr: A tuple (float) containing average PSNR over all slices.
    - mean_ssim : A tuple (float) containing average SSIM over all slices.
    - psnr_values : list (float) containing PSNR scores per sample.
    - ssim_values : list (float) containing SSIM scores per sample.
    """
    psnr_values, ssim_values = [], []

    for batch_idx, batch in enumerate(dataset):
        inputs = batch['input']
        targets = batch['target']

        preds = model.predict(inputs, verbose=0)

        for i in range(inputs.shape[0]):
            mid_depth = input_shape[2] // 2

            input_slice = inputs[i, :, :, mid_depth, 0].numpy()
            target_slice = targets[i, :, :, mid_depth, 0].numpy()
            pred_slice = preds[i, :, :, mid_depth, 0]

            psnr = peak_signal_noise_ratio(target_slice, pred_slice, data_range=1.0)
            ssim = structural_similarity(target_slice, pred_slice, data_range=1.0)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if batch_idx == 0 and i == 0:
                visualize_results(input_slice, target_slice, pred_slice, psnr, ssim)

    return np.mean(psnr_values), np.mean(ssim_values), psnr_values, ssim_values

# Evaluation Functions
def visualize_results(input_img, target_img, pred_img, psnr=None, ssim=None):
    """Visualize input, target and prediction slices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ["Input", "Target", "Prediction"]
    if psnr is not None and ssim is not None:
        titles[2] += f"\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}"
    
    images = [input_img, target_img, pred_img]
    
    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# Augmentation methods

class RotateAugmentation(layers.Layer):
    """
    A custom Keras layer to apply fixed-angle rotation augmentation to 3D medical images.

    This layer uses TensorFlow Addons to apply a specified degree of rotation to each image
    slice or volume.

    where:

    degrees : float, optional
        The fixed angle in degrees by which to rotate the input (default is 30°).
    **kwargs : dict
        Additional keyword arguments passed to the base Keras Layer class.
    """

    def __init__(self, degrees=30, **kwargs):
        super().__init__(**kwargs)
        self.radians = math.radians(degrees)

    def call(self, inputs):
        return tfa.image.rotate(
            inputs,
            self.radians,
            interpolation='bilinear',
            fill_mode='reflect'
        )


class CropAugmentation(layers.Layer):
    """
    A custom Keras layer that applies fixed center cropping and resizing to 3D volumes.

    This layer performs a center crop on a 5D input tensor (batch, height, width, depth, channels),
    followed by resizing of the spatial dimensions (height and width) back to the original size.
    Useful for enforcing spatial consistency during augmentation in 3D medical imaging.

    Inputs
    ----------
    - crop_size : tuple of int, the desired crop size as (height, width, depth, channels).
    - original_size : tuple of int, the original spatial size to which the volume is resized: (height, width).
    **kwargs : dict, additional keyword arguments passed to the base Layer class.

    Return
    -------
    tf.Tensor, the cropped and resized 5D tensor of shape (batch, height, width, depth, channels).
    """

    def __init__(self, crop_size, original_size, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.original_size = original_size

    def call(self, inputs):
        # Compute center offsets for cropping
        offsets = [
            (self.original_size[i] - self.crop_size[i]) // 2
            for i in range(3)
        ]

        # Crop the volume (batch, H, W, D, C)
        cropped = inputs[
            :,
            offsets[0]:offsets[0] + self.crop_size[0],
            offsets[1]:offsets[1] + self.crop_size[1],
            offsets[2]:offsets[2] + self.crop_size[2],
            :
        ]

        # Flatten depth and channels for resizing (batch, H, W, D * C)
        reshaped = tf.reshape(
            cropped,
            [tf.shape(cropped)[0], self.crop_size[0], self.crop_size[1], self.crop_size[2] * self.crop_size[3]]
        )

        # Resize height and width
        resized = tf.image.resize(
            reshaped,
            size=(self.original_size[0], self.original_size[1])
        )

        # Restore to original 5D shape (batch, H, W, D, C)
        restored = tf.reshape(
            resized,
            [tf.shape(cropped)[0], self.original_size[0], self.original_size[1], self.crop_size[2], self.crop_size[3]]
        )

        return restored


class IntensityAugmentation(layers.Layer):
    """
    A custom Keras layer to apply fixed brightness and contrast adjustments.

    This layer applies deterministic brightness and contrast changes to 3D volumes
    (typically with shape [batch, height, width, depth, channels]) using TensorFlow's
    image processing utilities. It is primarily useful for testing model robustness
    under fixed intensity variations.

    Inputs
    ----------
    - brightness : float (optional), brightness offset to apply (default is 0.1).
    - contrast : float (optional), contrast scaling factor (default is 3.0).
    **kwargs : dict, additional keyword arguments for the base Layer class.

    Return
    -------
    tf.Tensor, intensity-adjusted tensor with the same shape as the input.
    """

    def __init__(self, brightness=0.1, contrast=3.0, **kwargs):
        super().__init__(**kwargs)
        self.brightness = brightness
        self.contrast = contrast

    def call(self, inputs):
        x = tf.image.adjust_brightness(inputs, self.brightness)
        return tf.image.adjust_contrast(x, self.contrast)


class FlipAugmentation(layers.Layer):
    """
    A custom Keras layer that applies fixed vertical and horizontal flipping.

    This layer reverses the input tensor along the height (axis 1) and width (axis 2)
    dimensions, effectively performing a vertical and horizontal flip.

    This augmentation is useful for increasing data variability and testing model
    robustness to spatial orientation.

    Return
    -------
    tf.Tensor, the flipped tensor with the same shape as the input.
    """

    def call(self, inputs):
        x = tf.reverse(inputs, axis=[1])  # Flip vertically
        return tf.reverse(x, axis=[2])    # Flip horizontally



def apply_augmentation(input_tensor,
                       apply_rotate=False,
                       apply_crop=False,
                       apply_intensity=False,
                       apply_flip=False):
    """
    Apply selected fixed augmentations to a 3D input tensor using a sequential pipeline.

    Inputs
    ----------
    - input_tensor : tf.Tensor, the input 5D tensor with shape (batch, height, width, depth, channels).
    - apply_rotate : bool (ptional), whether to apply fixed-angle rotation (default is False).
    - apply_crop : bool (optional), whether to apply fixed center cropping and resizing (default is False).
    - apply_intensity : bool (optional), whether to apply fixed brightness and contrast adjustment (default is False).
    - apply_flip : bool (optional), whether to apply fixed vertical and horizontal flipping (default is False).

    Return
    -------
    tf.Tensor, the augmented tensor after applying the selected transformations.
    """
    layers_list = []

    if apply_rotate:
        layers_list.append(RotateAugmentation(degrees=30))

    if apply_crop:
        crop_size = (100, 100, 64, 1)
        original_size = (128, 128, 64, 1)
        layers_list.append(CropAugmentation(crop_size=crop_size, original_size=original_size))

    if apply_intensity:
        layers_list.append(IntensityAugmentation(brightness=0.1, contrast=3))

    if apply_flip:
        layers_list.append(FlipAugmentation())

    augmentation_pipeline = tf.keras.Sequential(layers_list)
    return augmentation_pipeline(input_tensor)

def ensure_5d(input_tensor):
    """
    Ensure the input tensor has 5 dimensions: (batch, depth, height, width, channels).

    Supported input shapes:
    - 3D (depth, height, width)         -> expands to (1, depth, height, width, 1)
    - 4D (batch, depth, height, width)  -> expands to (batch, depth, height, width, 1)
    - 5D (batch, depth, height, width, channels) -> returned as-is

    Parameters
    ----------
    input_tensor : tf.Tensor
        The input tensor to validate and expand if necessary.

    Returns
    -------
    tf.Tensor
        A 5D tensor with shape (batch, depth, height, width, channels).
    """
    if len(input_tensor.shape) == 3:
        return tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=-1)
    elif len(input_tensor.shape) == 4:
        return tf.expand_dims(input_tensor, axis=-1)
    elif len(input_tensor.shape) == 5:
        return input_tensor
    else:
        raise ValueError(f"Unsupported tensor shape {input_tensor.shape}: Expected 3D, 4D, or 5D.")

