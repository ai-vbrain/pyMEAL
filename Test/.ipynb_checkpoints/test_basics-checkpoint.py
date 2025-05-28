__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import unittest
import tensorflow as tf
import numpy as np
import tempfile
import os
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch
import sys
project_path = os.path.abspath("..")  
sys.path.append(project_path)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import pyMEAL.basics as bcs
import pyMEAL.builder_block as builder
import tempfile



# Constants
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 1

import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Non-interactive backend

class TestBasicsModule(unittest.TestCase):

    def test_apply_windowing_basic_range(self):
        data = np.linspace(-100, 100, 1000).reshape((10, 10, 10))
        result = bcs.apply_windowing(data, window_level=0, window_width=200)
        self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_resize_volume_shape(self):
        vol = np.random.rand(32, 32, 32)
        resized = bcs.resize_volume(vol, TARGET_SHAPE)
        self.assertEqual(resized.shape, TARGET_SHAPE)

    def test_normalize_volume(self):
        vol = np.random.rand(*TARGET_SHAPE) * 100
        normed = bcs.normalize_volume(vol)
        self.assertAlmostEqual(normed.min(), 0.0, places=3)
        self.assertAlmostEqual(normed.max(), 1.0, places=3)

    def test_ensure_5d(self):
        for shape in [TARGET_SHAPE, (1, *TARGET_SHAPE), (BATCH_SIZE, *TARGET_SHAPE, 1)]:
            tensor = tf.random.uniform(shape)
            out = bcs.ensure_5d(tensor)
            self.assertEqual(len(out.shape), 5)

    def test_rotate_augmentation_layer(self):
        # Use only a single 2D slice to reduce to 4D input for tfa.image.rotate
        tensor = tf.random.uniform((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[-1]))  # (1, H, W, C)
        layer = bcs.RotateAugmentation(degrees=30)
        output = layer(tensor)
        self.assertEqual(output.shape, tensor.shape)

    def test_crop_augmentation_layer(self):
        tensor = tf.random.uniform((BATCH_SIZE, *INPUT_SHAPE))
        layer = bcs.CropAugmentation((100, 100, 64, 1), INPUT_SHAPE)
        self.assertEqual(layer(tensor).shape, tensor.shape)

    def test_intensity_augmentation_layer(self):
        tensor = tf.random.uniform((BATCH_SIZE, *INPUT_SHAPE))
        layer = bcs.IntensityAugmentation()
        self.assertEqual(layer(tensor).shape, tensor.shape)

    def test_flip_augmentation_layer(self):
        tensor = tf.random.uniform((BATCH_SIZE, *INPUT_SHAPE))
        layer = bcs.FlipAugmentation()
        self.assertEqual(layer(tensor).shape, tensor.shape)

    def test_apply_augmentation_combined(self):
        tensor = tf.random.uniform((BATCH_SIZE, *INPUT_SHAPE))
        # Override: use only one 2D slice for rotate
        tensor_4d = tensor[:, :, :, 0, :]  # shape (1, H, W, C)
        rotated = bcs.RotateAugmentation(degrees=30)(tensor_4d)
        self.assertEqual(rotated.shape, tensor_4d.shape)

        # Now test the rest in 5D as expected
        result = bcs.apply_augmentation(
            tensor,
            apply_crop=True,
            apply_intensity=True,
            apply_flip=True
        )
        self.assertEqual(result.shape, tensor.shape)

    def test_load_and_preprocess_shape_consistency(self):
        dummy_vol = np.random.rand(*TARGET_SHAPE)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.nii.gz")
            target_path = os.path.join(tmpdir, "target.nii.gz")

            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), input_path)
            nib.save(nib.Nifti1Image(dummy_vol, affine=np.eye(4)), target_path)

            input_tensor, target_tensor = bcs.load_and_preprocess(
                (tf.constant(input_path.encode()), tf.constant(target_path.encode())),
                target_shape=TARGET_SHAPE
            )
            self.assertEqual(input_tensor.shape, (*TARGET_SHAPE, 1))
            self.assertEqual(target_tensor.shape, (*TARGET_SHAPE, 1))

    def test_create_dataset_pipeline(self):
        dummy_vol = np.random.rand(*TARGET_SHAPE)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                vol = nib.Nifti1Image(dummy_vol, affine=np.eye(4))
                nib.save(vol, os.path.join(tmpdir, f"input_{i}.nii.gz"))
                nib.save(vol, os.path.join(tmpdir, f"target_{i}.nii.gz"))

            dataset = bcs.create_dataset(
                tmpdir,
                tmpdir,
                input_shape=INPUT_SHAPE,
                target_shape=TARGET_SHAPE,
                batch_size=1
            )
            for batch in dataset.take(1):
                self.assertEqual(batch[0].shape, (1, *INPUT_SHAPE))
                self.assertEqual(batch[1].shape, (1, *INPUT_SHAPE))

    def test_refined_residual_block_output_shape(self):
        inputs = tf.random.normal((BATCH_SIZE, *INPUT_SHAPE))
        output = bcs.refined_residual_block(inputs, filters=INPUT_SHAPE[-1])
        self.assertEqual(output.shape, inputs.shape)

    def test_evaluate_model(self):
        class IdentityModel(tf.keras.Model):
            def call(self, inputs):
                return tf.identity(inputs)

        dummy_input = tf.ones((BATCH_SIZE, *INPUT_SHAPE), dtype=tf.float32)
        dummy_target = tf.ones((BATCH_SIZE, *INPUT_SHAPE), dtype=tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices({
            'input': dummy_input,
            'target': dummy_target
        }).batch(BATCH_SIZE)

        model = IdentityModel()
        psnr, ssim, psnr_list, ssim_list = bcs.evaluate_model(model, dataset, INPUT_SHAPE)
        self.assertGreater(psnr, 30)
        self.assertGreater(ssim, 0.9)
        self.assertEqual(len(psnr_list), BATCH_SIZE)
        self.assertEqual(len(ssim_list), BATCH_SIZE)

    def test_ssim_loss_perfect_prediction(self):
        y_true = tf.ones((1, 128, 128, 64, 1))
        y_pred = tf.ones((1, 128, 128, 64, 1))
        loss = bcs.ssim_loss(y_true, y_pred)
        self.assertAlmostEqual(float(loss.numpy()), 0.0, places=5)

    def test_ssim_loss_incorrect_prediction(self):
        y_true = tf.ones((1, 64, 64, 32, 1))
        y_pred = tf.zeros_like(y_true)
        loss = bcs.ssim_loss(y_true, y_pred)
        self.assertGreater(loss.numpy(), 0.5)

    def test_combined_loss_perfect_prediction(self):
        y_true = tf.ones((1, 64, 64, 32, 1))
        y_pred = tf.ones_like(y_true)
        loss = bcs.combined_loss(y_true, y_pred)
        self.assertAlmostEqual(float(loss.numpy()), 0.0, places=5)

    def test_combined_loss_value_range(self):
        y_true = tf.ones((1, 64, 64, 32, 1))
        y_pred = tf.zeros_like(y_true)
        loss = bcs.combined_loss(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
        self.assertLessEqual(loss.numpy(), 2.0)
if __name__ == "__main__":
    unittest.main()

# 


class TestMultistreamIntegration(unittest.TestCase):

    def test_build_multistream_model_with_real_image(self):
        # Path to real image files
        ct_path = './CTScan data/processed_data/proceed_CT/te/sub-OAS30001_sess-d3132_CT.nii.gz'
        t1_path = './CTScan data/processed_data/proceed_T1/te/sub-OAS30001_ses-d3132_T1w_be.nii.gz'
        
        # Load and process
        ct_img = nib.load(ct_path).get_fdata()
        t1_img = nib.load(t1_path).get_fdata()

        # Resize and normalize
        input_vol = bcs.normalize_volume(bcs.resize_volume(ct_img, TARGET_SHAPE))
        target_vol = bcs.normalize_volume(bcs.resize_volume(t1_img, TARGET_SHAPE))

        input_vol = np.expand_dims(input_vol, axis=-1).astype(np.float32)
        target_vol = np.expand_dims(target_vol, axis=-1).astype(np.float32)

        # Build dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'input': tf.convert_to_tensor([input_vol]),
            'target': tf.convert_to_tensor([target_vol])
        }).batch(BATCH_SIZE)

        # Build model
        model = builder.build_multistream_model(INPUT_SHAPE)
        output = model(tf.convert_to_tensor([input_vol]))

        # Check output shape and evaluation metrics
        self.assertEqual(output.shape, (1, *INPUT_SHAPE))
        psnr, ssim, _, _ = bcs.evaluate_model(model, dataset, INPUT_SHAPE)

        # Assert on quality (initial weights may not give good metrics, so assert existence not quality)
        self.assertIsNotNone(psnr)
        self.assertIsNotNone(ssim)

if __name__ == "__main__":
    unittest.main()