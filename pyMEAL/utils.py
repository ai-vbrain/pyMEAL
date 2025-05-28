__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '26th May, 2025'



from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from statistics import mean
import tensorflow as tf
from . import basics as pr
import nibabel as nib
import os


def assessing_performance(model, dataset, input_shape):
    """
    Evaluate model performance using PSNR and SSIM metrics.

    Inputs:
    ----------
    - model : keras.Model, the trained Keras model to evaluate.
    - dataset : tf.data.Dataset, dataset providing batches of input streams and target images.
    - input_shape : tuple, shape of the input images.

    Returns:
    -------
    - avg_psnr : float, the average PSNR across all evaluated slices.
    - avg_ssim : float, the average SSIM across all evaluated slices.
    - psnr_values : List of PSNR values for each evaluated slice.
    - ssim_values : List of SSIM values for each evaluated slice.
    """
    psnr_values = []
    ssim_values = []

    for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating Batches")):
        # Unpack batch
        if isinstance(batch, (tuple, list)):
            inputs_wrapper, targets = batch
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # Validate and prepare model inputs
        if isinstance(inputs_wrapper, (tuple, list)) and len(inputs_wrapper) == 4:
            model_inputs = list(inputs_wrapper)
        else:
            raise ValueError(
                f"Expected a tuple/list of 4 tensors, got {type(inputs_wrapper)} with length {len(inputs_wrapper)}"
            )

        # Run model prediction
        predictions = model.predict(model_inputs, verbose=0)

        batch_size = model_inputs[0].shape[0]
        for i in tqdm(range(batch_size), desc=f"Processing Batch {batch_idx} Samples", leave=False):
            mid_depth = model_inputs[0].shape[3] // 2

            input_slice = model_inputs[0][i, :, :, mid_depth, 0].numpy()
            target_slice = targets[i, :, :, mid_depth, 0].numpy()
            pred_slice = predictions[i, :, :, mid_depth, 0]

            # Calculate PSNR and SSIM
            psnr = peak_signal_noise_ratio(target_slice, pred_slice, data_range=1.0)
            ssim = structural_similarity(target_slice, pred_slice, data_range=1.0)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if batch_idx == 0 and i == 0:
                print(f"First sample PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")

    # Compute averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    print(f"\nFinal Average PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim, psnr_values, ssim_values


def compute_psnr_ssim(predictions, prediction_names, ground_truth, original_image, batch_index=0, save_dir=None, visualize=False):
    """
    Compute and optionally visualize PSNR/SSIM for a list of predictions.

    Inputs
    ----------
    - predictions : list of np.ndarray, containing predicted images.
    - prediction_names : list of str (names for each prediction).
    - ground_truth : np.ndarray, Ground truth image.
    - original_image : np.ndarray, normalized ground truth image.
    - batch_index : int, Batch index to process.
    - save_dir : str or None, Directory to save figures. If None, figures are not saved.
    - visualize : bool, Whether to display figures.

    Returns
    -------
    - psnr_scores : list of float, PSNR values for each prediction.
    - ssim_scores : list of float, SSIM values for each prediction.
    """
    psnr_scores = []
    ssim_scores = []

    for idx, (pred, name) in enumerate(zip(predictions, prediction_names)):
        # Extract batch slice if necessary
        if pred.ndim == 5:
            pred = np.squeeze(pred[batch_index], axis=-1)
        elif pred.ndim == 4:
            pred = pred[batch_index]

        # Prepare figure
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(pred[:, :, pred.shape[2] // 2], cmap='gray')
        plt.title(f'{name} - Prediction')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(ground_truth[:, :, ground_truth.shape[2] // 2], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        # Save if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{name.replace(" ", "_")}.png')
            plt.savefig(save_path, bbox_inches='tight')

        # Display or close based on flag
        if visualize:
            plt.show()
        else:
            plt.close()

        # Compute PSNR and SSIM
        gt_slice = ground_truth[:, :, ground_truth.shape[2] // 2]
        pred_slice = pred[:, :, pred.shape[2] // 2]
        psnr_val = psnr(gt_slice, pred_slice, data_range=1.0)
        ssim_val = ssim(gt_slice, pred_slice, data_range=1.0)

        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)

    return psnr_scores, ssim_scores


def tf_load_and_preprocess(input_path, target_path, target_shape):
    """
    TensorFlow wrapper for loading and preprocessing NIfTI volumes.
    
    Inputs
    ----------
    - input_path: Path to the input NIfTI file (as Tensor).
    - target_path: Path to the target NIfTI file (as Tensor).
    - target_shape: Target volume shape for resizing.

    Return
    ----------
    A tuple of (input_volume, target_volume) as TensorFlow tensors.
    """
    input_vol, target_vol = tf.py_function(
        func=lambda inp, tar: load_and_preprocess(inp, tar, target_shape),
        inp=[input_path, target_path],
        Tout=[tf.float32, tf.float32]
    )
    input_vol.set_shape(target_shape + (1,))
    target_vol.set_shape(target_shape + (1,))
    return input_vol, target_vol


def load_and_preprocess(input_path, target_path, target_shape):
    """
    Load and preprocess NIfTI volumes.

    Inputs
    ----------
    - input_path: Numpy-decoded path to input NIfTI file.
    - target_path: Numpy-decoded path to target NIfTI file.
    - target_shape: Target volume shape for resizing.

    Return
    ----------
    A tuple of preprocessed (input_volume, target_volume).
    """

    def apply_windowing(data, window_level=40, window_width=80):
        lower = window_level - window_width / 2
        upper = window_level + window_width / 2
        data = np.clip(data, lower, upper)
        return (data - lower) / (upper - lower)

    input_vol = nib.load(input_path.numpy().decode('utf-8')).get_fdata()
    input_vol = apply_windowing(input_vol)

    target_vol = nib.load(target_path.numpy().decode('utf-8')).get_fdata()

    input_vol = pr.resize_volume(input_vol, target_shape)
    target_vol = pr.resize_volume(target_vol, target_shape)

    input_vol = np.expand_dims(input_vol, -1).astype(np.float32)
    target_vol = np.expand_dims(target_vol, -1).astype(np.float32)

    input_vol = (input_vol - np.min(input_vol)) / (np.max(input_vol) - np.min(input_vol))
    target_vol = (target_vol - np.min(target_vol)) / (np.max(target_vol) - np.min(target_vol))

    return input_vol, target_vol


def create_dataset(input_dir, target_dir, target_shape, batch_size):
    """
    Create a TensorFlow data pipeline for loading and preparing NIfTI volume datasets.

    Inputs
    ----------
    - input_dir (str): Directory containing input NIfTI (.nii.gz) files.
    - target_dir (str): Directory containing target NIfTI (.nii.gz) files.
    - target_shape (tuple): Desired output volume shape, e.g., (128, 128, 64).
    - batch_size (int): Batch size for the dataset.

    Return
    ----------
    tf.data.Dataset: A preprocessed, batched, and prefetched TensorFlow dataset 
                         yielding ((input1, input2, input3, input4), target) pairs,
                         where each input is a duplicated version of the input volume.
    """

    # Collect sorted file paths
    import os
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.nii.gz')])

    # Create dataset from file pairs
    dataset = tf.data.Dataset.from_tensor_slices((input_files, target_files))

    # Map the preprocessing function with target shape
    dataset = dataset.map(
        lambda inp, tar: tf_load_and_preprocess(inp, tar, target_shape),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Duplicate the input volume into four streams for multi-stream models
    def duplicate_inputs(input_volume, target_volume):
        return (input_volume, input_volume, input_volume, input_volume), target_volume

    dataset = dataset.map(duplicate_inputs, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compute_psnr(volume_true, volume_pred, max_val=1.0):
    

    """
    Computes the mean PSNR and its standard deviation across all slices of a 3D volume.

    Inputs
    ----------
    - volume_true (tf.Tensor): Ground truth 3D volume tensor of shape (H, W, D).
    - volume_pred (tf.Tensor): Predicted 3D volume tensor of shape (H, W, D).
    - max_val (float): The maximum possible pixel value.

    Return
    ----------
    - mean PSNR and its std (tuple): (mean_psnr, psnr_std) where mean_psnr is the average PSNR across slices,
               and psnr_std is the standard deviation of these PSNR values.
    """

    psnr_values = []
    volume_true = tf.convert_to_tensor(volume_true, dtype=tf.float32)
    volume_pred = tf.convert_to_tensor(volume_pred, dtype=tf.float32)

    for i in range(volume_true.shape[2]):
        slice_true = tf.expand_dims(volume_true[:, :, i], axis=-1)
        slice_pred = tf.expand_dims(volume_pred[:, :, i], axis=-1)
        psnr = tf.image.psnr(slice_true, slice_pred, max_val=max_val).numpy()
        psnr = psnr if np.isfinite(psnr) else 100.0  # Cap infinite PSNR
        psnr_values.append(psnr)

    psnr_values = np.array(psnr_values)
    return np.mean(psnr_values), np.std(psnr_values)


def compute_ssim(volume_true, volume_pred, max_val=1.0):

    """
    Computes the mean SSIM and its standard deviation across all slices of a 3D volume.

    Inputs
    ----------
    - volume_true (tf.Tensor): Ground truth 3D volume tensor of shape (H, W, D).
    - volume_pred (tf.Tensor): Predicted 3D volume tensor of shape (H, W, D).
    - max_val (float): The maximum possible pixel value.

    Return
    ----------
    mean ssim and its std (tuple): (mean_ssim, ssim_std) where mean_ssim is the average SSIM across slices,
               and ssim_std is the standard deviation of these SSIM values.
    """


    ssim_values = []
    volume_true = tf.convert_to_tensor(volume_true, dtype=tf.float32)
    volume_pred = tf.convert_to_tensor(volume_pred, dtype=tf.float32)

    for i in range(volume_true.shape[2]):
        slice_true = tf.expand_dims(volume_true[:, :, i], axis=-1)
        slice_pred = tf.expand_dims(volume_pred[:, :, i], axis=-1)
        ssim = tf.image.ssim(slice_true, slice_pred, max_val=max_val).numpy()
        ssim_values.append(ssim)

    ssim_values = np.array(ssim_values)
    return np.mean(ssim_values), np.std(ssim_values)



def plot_translated_image(model, dataset, visualize=True):
    """
    Evaluate a model's 3D predictions using PSNR and SSIM metrics.

    Inputs
    ----------
    - model : keras.Model, the trained Keras model to evaluate.
    - dataset : tf.data.Dataset, dataset providing batches of input streams and target images.
    - visualize : bool, (optional) Whether to visualize the middle slice and difference maps, by default True.

    Returns
    -------
    - avg_psnr and std (tuple):  average PSNR and its standard deviation.
    - avg_ssim and std (tuple): average SSIM and its standard deviation.
    - Lists of all individual PSNR and SSIM values.
    """
    psnr_values = []
    ssim_values = []
    psnr_stds = []
    ssim_stds = []

    for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating Batches")):
        input_streams, targets = batch

        # Handle tuple input_streams
        if isinstance(input_streams, tuple):
            input_streams = tf.stack(input_streams, axis=-1)

        # Split the input tensor into 4 separate tensors
        if input_streams.shape[-1] == 4:
            input_list = tf.split(input_streams, num_or_size_splits=4, axis=-1)
            input_list = [tf.squeeze(x, axis=-1) for x in input_list]
        else:
            raise ValueError("Input tensor does not have 4 components along the last axis.")

        # Model prediction
        predictions = model.predict(input_list, verbose=0)

        # Handle different prediction output types
        if isinstance(predictions, dict):
            predictions = predictions['reconstructed_output']
        elif isinstance(predictions, list):
            predictions = predictions[0]

        batch_size = input_streams.shape[0]
        for i in tqdm(range(batch_size), desc=f"Processing Batch {batch_idx} Samples", leave=False):
            target_volume = targets[i].numpy()
            pred_volume = predictions[i]

            psnr, psnr_std = compute_psnr(target_volume, pred_volume)
            ssim, ssim_std = compute_ssim(target_volume, pred_volume)

            psnr_values.append(psnr)
            ssim_values.append(ssim)
            psnr_stds.append(psnr_std)
            ssim_stds.append(ssim_std)

            # Visualization of the first sample in the first batch
            if visualize and batch_idx == 0 and i == 0:
                mid_depth = input_streams.shape[3] // 2
                input_slice = input_streams[i, :, :, mid_depth, 0].numpy()
                target_slice = targets[i, :, :, mid_depth, 0].numpy()
                pred_slice = predictions[i, :, :, mid_depth, 0]

                # 
                plt.imshow(pred_slice, cmap='gray', vmin=0, vmax=1)
                plt.title(f"Predicted Slice\nPSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
                plt.axis("off")
                plt.tight_layout()

                fig = plt.gcf()
                fig.set_size_inches(7.472, 5)
                plt.show()

                diff_map = np.abs(target_slice - pred_slice)
                plt.imshow(diff_map, cmap='jet')
                plt.title("Difference Heatmap")
                plt.axis("off")

                plt.tight_layout()
                fig = plt.gcf()
                fig.set_size_inches(7.472, 5)
                plt.show()

                # 
                diff_values = (target_slice - pred_slice).flatten()
                plt.hist(diff_values, bins=50, color='red', alpha=0.7)
                plt.title("Pixel Difference Histogram")
                plt.xlabel("Difference Intensity")
                plt.ylabel("Frequency")

                fig = plt.gcf()
                fig.set_size_inches(7.472, 5)
                # plt.grid(True)  
                plt.show()

    # Summary statistics
    avg_psnr = mean(psnr_values)
    avg_ssim = mean(ssim_values)
    std_psnr = mean(psnr_stds)
    std_ssim = mean(ssim_stds)

    return (avg_psnr, std_psnr), (avg_ssim, std_ssim), psnr_values, ssim_values
