__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import unittest
import tensorflow as tf
import numpy as np
import importlib
from unittest.mock import patch
import pyMEAL.traditional_augmentation
importlib.reload(pyMEAL.traditional_augmentation)
from pyMEAL.traditional_augmentation import (
    FlipAugmentation,
    RotateAugmentation,
    CropAugmentation,
    IntensityAugmentation,
    create_dataset,
    refined_diffusion_model
)

class TestAugmentationLayersTA(unittest.TestCase):
    def setUp(self):
        self.input_tensor = tf.random.uniform((1, 128, 128, 64, 1))

    def test_flip_augmentation(self):
        layer = FlipAugmentation()
        output = layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_rotate_augmentation(self):
        layer = RotateAugmentation(axis=(1, 2))
        output = layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_crop_augmentation(self):
        layer = CropAugmentation(target_shape=(128, 128, 64, 1))
        output = layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)

    def test_intensity_augmentation(self):
        layer = IntensityAugmentation()
        output = layer(self.input_tensor)
        self.assertEqual(output.shape, self.input_tensor.shape)
if __name__ == "__main__":
    unittest.main()

class TestRefinedDiffusionModelTA(unittest.TestCase):
    def test_model_output_shape(self):
        model = refined_diffusion_model((128, 128, 64, 1))
        dummy_input = tf.random.uniform((1, 128, 128, 64, 1))
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 128, 128, 64, 1))
if __name__ == "__main__":
    unittest.main()

class TestDatasetPipelineTA(unittest.TestCase):
    @patch('pyMEAL.traditional_augmentation.bcs.create_dataset')
    def test_create_dataset_pipeline(self, mock_create_dataset):
        dummy_input = tf.random.uniform((1, 128, 128, 64, 1))
        dummy_target = tf.random.uniform((1, 128, 128, 64, 1))
        mock_ds = tf.data.Dataset.from_tensor_slices((dummy_input, dummy_target)).batch(1)

        mock_create_dataset.return_value = mock_ds

        aug_ds = create_dataset("fake_input_dir", "fake_target_dir", batch_size=1, shuffle=False)

        for input_vol, target_vol in aug_ds.take(1):
            self.assertEqual(input_vol.shape, (1, 128, 128, 64, 1))
            self.assertEqual(target_vol.shape, (1, 128, 128, 64, 1))
if __name__ == "__main__":
    unittest.main()

