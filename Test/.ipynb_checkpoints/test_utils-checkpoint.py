__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '22nd May, 2025'


# import numpy as np
# import tensorflow as tf
# from scipy.ndimage import zoom
import matplotlib
import sys
project_path = os.path.abspath("..")  
sys.path.append(project_path)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


import unittest
import numpy as np
import os
import tempfile
import nibabel as nib
import matplotlib.pyplot as plt
from unittest.mock import MagicMock

import pyMEAL.utils as viz
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Non-interactive backend





import unittest
import numpy as np
import os
import tempfile
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from unittest.mock import MagicMock
import pyMEAL.builder_block as BD

import pyMEAL.utils as viz
from pyMEAL import basics as pr


INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 1


MODEL_DIR = "./saved_models/"
# os.makedirs(LOG_DIR, exist_ok=True)  
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuration at the top of your script

load_model = os.path.join(MODEL_DIR, "builder1_mode1l1abW512_1_11211z1p1rt_.h5")
##
print("Checking file existence at:", load_model)
print("File exists:" , os.path.exists(load_model))

##
custom_objects = {
    'FlipAugmentation': BD.FlipAugmentation,
    'RotateAugmentation': BD.RotateAugmentation,
    'CropAugmentation': BD.CropAugmentation,
    'IntensityAugmentation': BD.IntensityAugmentation
}
##
model = tf.keras.models.load_model(load_model, custom_objects= custom_objects, compile=False)
plt.switch_backend('Agg')

# Constants
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 1

# Paths to real image files
ct_path = './CTScan data/processed_data/proceed_CT/te/sub-OAS30001_sess-d3132_CT.nii.gz'
t1_path = './CTScan data/processed_data/proceed_T1/te/sub-OAS30001_ses-d3132_T1w_be.nii.gz'

def load_real_images(ct_path, t1_path, target_shape):
    ct_img = nib.load(ct_path).get_fdata()
    t1_img = nib.load(t1_path).get_fdata()

    ct_img = pr.resize_volume(ct_img, target_shape)
    t1_img = pr.resize_volume(t1_img, target_shape)

    ct_img = (ct_img - np.min(ct_img)) / (np.max(ct_img) - np.min(ct_img))
    t1_img = (t1_img - np.min(t1_img)) / (np.max(t1_img) - np.min(t1_img))

    return ct_img.astype(np.float32), t1_img.astype(np.float32)


class TestUtilsUnit(unittest.TestCase):

    def test_compute_psnr(self):
        ct_img, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
        mean_val, std_val = viz.compute_psnr(tf.convert_to_tensor(t1_img), tf.convert_to_tensor(t1_img))
        self.assertGreater(mean_val, 30)
        self.assertAlmostEqual(std_val, 0.0, places=3)

    def test_compute_ssim(self):
        ct_img, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
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
            in_vol, tar_vol = viz.load_and_preprocess(tf.convert_to_tensor(in_path), tf.convert_to_tensor(tar_path), TARGET_SHAPE)
            self.assertEqual(in_vol.shape, (*TARGET_SHAPE, 1))
            self.assertEqual(tar_vol.shape, (*TARGET_SHAPE, 1))

    def test_tf_load_and_preprocess(self):
        dummy_vol = np.random.rand(*TARGET_SHAPE)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, 'input.nii.gz')
            tar_path = os.path.join(tmpdir, 'target.nii.gz')
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), in_path)
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), tar_path)
            in_tensor, tar_tensor = viz.tf_load_and_preprocess(tf.constant(in_path), tf.constant(tar_path), TARGET_SHAPE)
            self.assertEqual(in_tensor.shape, (*TARGET_SHAPE, 1))
            self.assertEqual(tar_tensor.shape, (*TARGET_SHAPE, 1))


class TestVisualizeIntegration(unittest.TestCase):

    def test_create_dataset_pipeline(self):
        dummy_vol = np.random.rand(*TARGET_SHAPE)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), os.path.join(tmpdir, f'in_{i}.nii.gz'))
                nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), os.path.join(tmpdir, f'tar_{i}.nii.gz'))
            dataset = viz.create_dataset(tmpdir, tmpdir, TARGET_SHAPE, batch_size=1)
            for batch in dataset.take(1):
                self.assertEqual(batch[0][0].shape, (1, *TARGET_SHAPE, 1))
                self.assertEqual(batch[1].shape, (1, *TARGET_SHAPE, 1))

    def test_assessing_performance_real_model(self):
        class DummyModel(tf.keras.Model):
            def call(self, inputs): return tf.identity(inputs[0])

        ct_img, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
        x = np.expand_dims(ct_img, axis=-1)[np.newaxis, ...]  # (1, H, W, D, 1)
        y = np.expand_dims(t1_img, axis=-1)[np.newaxis, ...]
        dataset = tf.data.Dataset.from_tensor_slices(((x, x, x, x), y)).batch(1)

        # model = DummyModel()
        avg_psnr, avg_ssim, _, _ = viz.assessing_performance(model, dataset, INPUT_SHAPE)
        self.assertGreater(avg_psnr, 30)
        self.assertGreater(avg_ssim, 0.9)

    def test_plot_translated_image_real(self):
        class DummyModel(tf.keras.Model):
            def call(self, inputs): return tf.identity(inputs[0])

        ct_img, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
        x = np.expand_dims(ct_img, axis=-1)[np.newaxis, ...]
        y = np.expand_dims(t1_img, axis=-1)[np.newaxis, ...]
        dataset = tf.data.Dataset.from_tensor_slices(((x, x, x, x), y)).batch(1)

        model = DummyModel()
        (avg_psnr, std_psnr), (avg_ssim, std_ssim), _, _ = viz.plot_translated_image(model, dataset, visualize=False)
        self.assertGreater(avg_psnr, 30)
        self.assertGreater(avg_ssim, 0.9)

    def test_compute_psnr_ssim_visualize(self):
        ct_img, t1_img = load_real_images(ct_path, t1_path, TARGET_SHAPE)
        pred = np.expand_dims(ct_img, axis=(0, -1))  # shape (1, H, W, D, 1)
        truth = t1_img  # shape (H, W, D)
        norm = (truth - np.min(truth)) / (np.max(truth) - np.min(truth))
        psnrs, ssims = viz.compute_psnr_ssim([pred], ['Dummy'], truth, norm, visualize=False)
        self.assertEqual(len(psnrs), 1)
        self.assertEqual(len(ssims), 1)


# Run tests
if __name__ == '__main__':
    plt.switch_backend('Agg') 
    unittest.main()
# unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestMultistreamIntegration))
