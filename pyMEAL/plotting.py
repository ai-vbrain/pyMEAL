__authors__ = 'Abdulmojeed Ilyas, and Adeleke Maradesa'

__date__ = '21st May, 2025'

# laod necessary supporting library

import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean
from scipy.stats import sem
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


input_shape = (128, 128, 64, 1)
target_shape = input_shape[:3] 
batch_size = 1

def psnr3d(y_true, y_pred, max_val=1.0):
    psnr_vals = []
    for i in range(y_true.shape[2]):  # depth axis
        psnr_vals.append(tf.image.psnr(y_true[:, :, i], y_pred[:, :, i], max_val=max_val))
    return tf.reduce_mean(psnr_vals)

def ssim3d(y_true, y_pred, max_val=1.0):
    ssim_vals = []
    for i in range(y_true.shape[2]):  # depth axis
        ssim_vals.append(tf.image.ssim(y_true[:, :, i], y_pred[:, :, i], max_val=max_val))
    return tf.reduce_mean(ssim_vals)


def ssim(volume_1, volume_2):
    """
    Computes the SSIM (Structural Similarity Index) and its standard deviation between two 3D volumes.

    This function converts the input volumes to float32, computes voxel-wise SSIM using `ssim3d`,
    and returns the mean and standard deviation of the SSIM values.

    Inputs:
    ----------
    - volume_1 : tf.Tensor, the first 3D image tensor (e.g., predicted volume).
    - volume_2 : tf.Tensor, the second 3D image tensor (e.g., ground truth volume).

    Return:
    -------
    -  mean_ssim : (float) Mean SSIM between the two volumes.
    -  std_ssim : (float) Standard deviation of the SSIM values.
    """
    volume_1 = tf.image.convert_image_dtype(volume_1, tf.float32)
    volume_2 = tf.image.convert_image_dtype(volume_2, tf.float32)
    ssim_values = ssim3d(
        volume_1, volume_2,
        max_val=1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        name='ssim3d'
    )
    mean_ssim = np.mean(ssim_values)
    std_ssim = np.std(ssim_values)
    return mean_ssim, std_ssim


def psnr(volume_1, volume_2):
    """
    Computes the PSNR (Peak Signal-to-Noise Ratio) and its standard deviation between two 3D volumes.

    This function converts the input volumes to float32, computes voxel-wise PSNR using `psnr3d`,
    and returns the mean and standard deviation of the PSNR values.

    Inputs:
    ----------
    - volume_1 : tf.Tensor, the first 3D image tensor (e.g., predicted volume).
    - volume_2 : tf.Tensor, the second 3D image tensor (e.g., ground truth volume).

    Returns:
    -------
    - mean_psnr : (float) Mean PSNR between the two volumes.
    - std_psnr : (float) Standard deviation of the PSNR values.
    """
    volume_1 = tf.image.convert_image_dtype(volume_1, tf.float32)
    volume_2 = tf.image.convert_image_dtype(volume_2, tf.float32)
    psnr_values = psnr3d(volume_1, volume_2, max_val=1.0, name='psnr3d')
    mean_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    return mean_psnr, std_psnr


def visualize_plot(model, dataset, aug_type = 'FL', visualize=True, view_type='CT_MRI'):
    ##
    """
    Evaluate the model's performance on a validation dataset using PSNR and SSIM metrics.

    Inputs
    ----------
    - model : keras.Model, the trained model to evaluate.
    - dataset : tf.data.Dataset, dataset yielding input streams and target volumes.
    - visualize : bool, (optional) If True, visualizes the first sample's input, target, and prediction slices.
    - view_type : str, (optional) 'CT_MRI' to show mid-slice only, 'all' to show all slices with PSNR/SSIM annotations.
    - aug_type : str (optional), augmentation type identifier. Use 'FL' for multi-stream input models,
        and any other value for single-stream models.

    Returns
    -------
    tuple
    - avg_psnr : (float) The average Peak Signal-to-Noise Ratio across all samples.
    - avg_ssim : (float) The average Structural Similarity Index across all samples.
    - psnr_values : List of PSNR values for each sample.
    - ssim_values : List of SSIM values for each sample.
  
    """
    psnr_values = []
    ssim_values = []

    for batch_idx, batch in enumerate(tqdm(dataset, desc="Processing Batches")):
        input_data, targets = batch

        if aug_type == 'FL':
            predictions = model.predict(input_data)
            visual_input = input_data[0]  # Only visualize the first stream (flip)
        else:
            predictions = model.predict(input_data)
            visual_input = input_data

        for i in tqdm(range(visual_input.shape[0]), desc=f"Batch {batch_idx} Samples", leave=False):
            mid_depth = visual_input.shape[3] // 2

            input_slice = visual_input[i, :, :, mid_depth, 0].numpy()
            target_slice = targets[i, :, :, mid_depth, 0].numpy()
            pred_slice = predictions[i, :, :, mid_depth, 0]

            data_range = target_slice.max() - target_slice.min()
            psnr = peak_signal_noise_ratio(target_slice, pred_slice, data_range=data_range)
            ssim = structural_similarity(target_slice, pred_slice, data_range=data_range)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if visualize and batch_idx == 0 and i == 0:
                if view_type == "CT_MRI":
                    _visualize_slices(input_slice, target_slice, pred_slice, psnr, ssim)
                elif view_type == "all":
                    input_vol = visual_input[i, :, :, :, 0].numpy()
                    target_vol = targets[i, :, :, :, 0].numpy()
                    pred_vol = predictions[i, :, :, :, 0]
                    visualize_all_slices(input_vol, target_vol, pred_vol)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    return avg_psnr, avg_ssim, psnr_values, ssim_values


def _visualize_slices(input_slice, target_slice, pred_slice, psnr, ssim):
    """
    Visualize the input, target, and predicted slices along with PSNR and SSIM metrics.
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(input_slice, cmap='gray')
    plt.title("Input Slice")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_slice, cmap='gray')
    plt.title(f"Predicted Slice\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_all_slices(input_volume, target_volume, pred_volume):
    """
    Visualize all slices from input and predicted volumes side by side with PSNR and SSIM metrics.

    Inputs
    -------
    input_volume : np.ndarray, the input volume as a 3D or 4D array. If 4D, the last dimension is assumed to be channel size 1.
    target_volume : np.ndarray, taget_volume as a 3D or 4D array. If 4D, the last dimension is assumed to be channel size 1.
    pred_volume : np.ndarray, the predicted volume as a 3D or 4D array. If 4D, the last dimension is assumed to be channel size 1.
    
    Return
    -------
    Show plot for each slice showing the input and prediction with PSNR and SSIM metrics
    """
    if input_volume.ndim == 4:
        input_volume = input_volume[..., 0]
    if target_volume.ndim == 4:
        target_volume = target_volume[..., 0]
    if pred_volume.ndim == 4:
        pred_volume = pred_volume[..., 0]

    num_slices = input_volume.shape[2]

    for slice_idx in range(num_slices):
        input_slice = input_volume[:, :, slice_idx]
        target_slice = target_volume[:, :, slice_idx]
        pred_slice = pred_volume[:, :, slice_idx]

        data_range = target_slice.max() - target_slice.min()
        psnr = peak_signal_noise_ratio(target_slice, pred_slice, data_range=data_range)
        ssim = structural_similarity(target_slice, pred_slice, data_range=data_range)

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(input_slice, cmap='gray')
        plt.title(f'Input Slice {slice_idx}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(pred_slice, cmap='gray')
        plt.title(f'Predicted Slice {slice_idx}\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def visualize_augmentation(original_data, augmented_data, slice_index=32, name='Augmented'):
    """
    Visualize and compare original and augmented data for a given slice.

    Inputs
    -------
    - original_data: numpy array (original 3D data)
    - augmented_data: numpy array (augmented 3D data)
    - slice_index: int, index of the slice to display
    - name: str, title for the augmented image

    Return
    -------
    original and augmented data
    """
    # Handle possible batch and channel dimensions
    if len(augmented_data.shape) == 4:
        augmented_data = np.squeeze(augmented_data, axis=(0, -1))
    elif len(augmented_data.shape) == 5:
        augmented_data = np.squeeze(augmented_data, axis=(0, -1))

    plt.figure(figsize=(10, 5))
    # Optional display of the original slice if provided
    if original_data is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(original_data[:, :, slice_index], cmap='gray')
        plt.title('Original Slice')
        plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_data[:, :, slice_index], cmap='gray')
    plt.title(f'{name} Slice')
    plt.axis('off')

    print(f'{name} Shape:', augmented_data.shape)
    plt.show()


def visualize_predictions_with_difference(pred_list, pred_names, ground_truth, original_data, batch_index=0):
    """
    Visualizes prediction slices, difference maps, and histograms side-by-side.
    
    Inputs
    -------
    - pred_list: List of prediction tensors [B, H, W, D, C] or [B, H, W, D]
    - pred_names: List of corresponding names for display
    - ground_truth: Ground truth tensor [H, W, D]
    - original_data: Original input tensor [H, W, D]
    - batch_index: Index of the batch to visualize (default 0)

    Return
    -------
    prediction slices, difference maps, and histograms plots
    """
    mid_depth = original_data.shape[2] // 2  # Middle slice
    target_slice = ground_truth[:, :, mid_depth]

    for pred, name in zip(pred_list, pred_names):
        # Extract slice for the specified batch and channel
        if pred.ndim == 5:
            pred_slice = pred[batch_index][:, :, mid_depth, 0]
        elif pred.ndim == 4:
            pred_slice = pred[batch_index][:, :, mid_depth]
        else:
            raise ValueError("Prediction must be 4D or 5D tensor")

        # Compute difference map
        diff_map = np.abs(target_slice - pred_slice)
        diff_values = (target_slice - pred_slice).flatten()

        # Plot prediction, difference map, and histogram
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Predicted Slice
        axes[0].imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"{name} - Prediction", fontsize=14)
        axes[0].axis("off")

        # Difference Map
        axes[1].imshow(diff_map, cmap='jet', vmin=0, vmax=0.5)
        axes[1].set_title(f"{name} - Difference Map", fontsize=14)
        axes[1].axis("off")

        # Histogram
        axes[2].hist(diff_values, bins=50, color='blue', alpha=0.7,
                     edgecolor='black', weights=np.ones_like(diff_values) / 1000)
        axes[2].set_xlim(-0.3, 0.3)
        axes[2].set_ylim(0, 10)
        axes[2].set_xlabel("Pixel Difference", fontsize=16)
        axes[2].set_ylabel("Frequency/1000", fontsize=16)
        axes[2].set_title(f"{name} - Histogram", fontsize=14)
        for spine in axes[2].spines.values():
            spine.set_linewidth(1)

        plt.tight_layout()
        plt.show()
