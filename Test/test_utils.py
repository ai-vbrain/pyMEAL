__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import os
import tempfile
#
import pyMEAL.builder_block as BD
import tensorflow as tf

import matplotlib 
import matplotlib.pyplot as plt
import sys
project_path = os.path.abspath("..")  
sys.path.append(project_path)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
from unittest.mock import patch
import nibabel as nib
import pyMEAL.basics as bcs
import unittest
import numpy as np
import pyMEAL.utils as viz


'''
This code section test the plotting and util modules
'''

# Constants and Paths
MODEL_DIR = "./saved_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

load_model_path = os.path.join(MODEL_DIR, "builder1_mode1l1abW512_1_11211z1p1rt_.h5")
print("Checking file existence at:", load_model_path)
print("File exists:", os.path.exists(load_model_path))

custom_objects = {
    'FlipAugmentation': BD.FlipAugmentation,
    'RotateAugmentation': BD.RotateAugmentation,
    'CropAugmentation': BD.CropAugmentation,
    'IntensityAugmentation': BD.IntensityAugmentation
}

model = tf.keras.models.load_model(load_model_path, custom_objects=custom_objects, compile=False)

import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Non-interactive backend

INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 1

# Real image paths
ct_path = './CTScan data/processed_data/proceed_CT/te/sub-OAS30001_sess-d3132_CT.nii.gz'
t1_path = './CTScan data/processed_data/proceed_T1/te/sub-OAS30001_ses-d3132_T1w_be.nii.gz'


def load_real_images(ct_path, t1_path, target_shape):
    ct_img = nib.load(ct_path).get_fdata()
    t1_img = nib.load(t1_path).get_fdata()

    ct_img = bcs.resize_volume(ct_img, target_shape)
    t1_img = bcs.resize_volume(t1_img, target_shape)

    ct_img = (ct_img - np.min(ct_img)) / (np.max(ct_img) - np.min(ct_img))
    t1_img = (t1_img - np.min(t1_img)) / (np.max(t1_img) - np.min(t1_img))

    return ct_img.astype(np.float32), t1_img.astype(np.float32)


# --- Unit Tests ---
class TestUtilsUnit(unittest.TestCase):

    def test_compute_psnr(self):
        _, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
        mean_val, std_val = viz.compute_psnr(tf.convert_to_tensor(t1_img), tf.convert_to_tensor(t1_img))
        self.assertGreater(mean_val, 30)
        self.assertAlmostEqual(std_val, 0.0, places=3)

    def test_compute_ssim(self):
        _, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
        mean_val, std_val = viz.compute_ssim(tf.convert_to_tensor(t1_img), tf.convert_to_tensor(t1_img))
        self.assertGreater(mean_val, 0.9)
        self.assertAlmostEqual(std_val, 0.0, places=3)

    def test_load_and_preprocess(self):
        dummy_vol = np.random.rand(*TARGET_SHAPE)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, 'input.nii.gz')
            tar_path = os.path.join(tmpdir, 'target.nii.gz')
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), in_path)
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), tar_path)
            in_vol, tar_vol = viz.load_and_preprocess(tf.convert_to_tensor(in_path),
                                                      tf.convert_to_tensor(tar_path),
                                                      TARGET_SHAPE)
            self.assertEqual(in_vol.shape, (*TARGET_SHAPE, 1))
            self.assertEqual(tar_vol.shape, (*TARGET_SHAPE, 1))

    def test_tf_load_and_preprocess(self):
        dummy_vol = np.random.rand(*TARGET_SHAPE)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, 'input.nii.gz')
            tar_path = os.path.join(tmpdir, 'target.nii.gz')
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), in_path)
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), tar_path)
            in_tensor, tar_tensor = viz.tf_load_and_preprocess(tf.constant(in_path),
                                                               tf.constant(tar_path),
                                                               TARGET_SHAPE)
            self.assertEqual(in_tensor.shape, (*TARGET_SHAPE, 1))
            self.assertEqual(tar_tensor.shape, (*TARGET_SHAPE, 1))
# Run Test
if __name__ == "__main__":
    unittest.main()