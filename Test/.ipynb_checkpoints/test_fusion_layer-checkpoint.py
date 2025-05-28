__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import os
import unittest
import tensorflow as tf
import numpy as np
import tempfile
import pyMEAL.basics as bcs
from pyMEAL.fusion_layer import create_dataset, build_multistream_model, FeatureFusionLayer


def save_nifti(filepath, volume):
    # Save volume as .npy instead
    np.save(filepath.replace('.nii.gz', '.npy'), volume.numpy() if hasattr(volume, 'numpy') else volume)

# Patch it manually
bcs.save_nifti = save_nifti
class TestCreateDatasetFL(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for input/target
        self.input_dir = tempfile.TemporaryDirectory()
        self.target_dir = tempfile.TemporaryDirectory()

        # Save dummy .nii.gz files using bcs.save_nifti
        for i in range(2):
            dummy = tf.random.uniform((128, 128, 64), dtype=tf.float32)
            bcs.save_nifti(os.path.join(self.input_dir.name, f"input_{i}.nii.gz"), dummy)
            bcs.save_nifti(os.path.join(self.target_dir.name, f"target_{i}.nii.gz"), dummy)

    def test_dataset_output_shapes(self):
        dataset = create_dataset(
            self.input_dir.name,
            self.target_dir.name,
            batch_size=1,
            shuffle=False
        )

        for batch in dataset.take(1):
            inputs, targets = batch
            self.assertIsInstance(inputs, dict)
            self.assertIn('flip_stream', inputs)
            self.assertEqual(inputs['flip_stream'].shape, (1, 128, 128, 64, 1))
            self.assertEqual(targets.shape, (1, 128, 128, 64, 1))

    def tearDown(self):
        self.input_dir.cleanup()
        self.target_dir.cleanup()
if __name__ == '__main__':
    unittest.main()


class TestFeatureFusionLayerFL(unittest.TestCase):
    def test_fusion_layer_output(self):
        batch_size = 2
        input_shape = (batch_size, 32, 32, 16, 8)
        inputs = [tf.random.normal(input_shape) for _ in range(4)]

        layer = FeatureFusionLayer(units=16)
        output = layer(inputs)

        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(output.shape, (batch_size, 32, 32, 16, 16))
if __name__ == '__main__':
    unittest.main()


class TestBuildMultistreamModelFL(unittest.TestCase):
    def test_forward_pass(self):
        input_shape = (128, 128, 64, 1)
        model = build_multistream_model(input_shape)

        dummy_input = {
            'flip_stream': tf.random.uniform((1, *input_shape)),
            'rotate_stream': tf.random.uniform((1, *input_shape)),
            'crop_stream': tf.random.uniform((1, *input_shape)),
            'intensity_stream': tf.random.uniform((1, *input_shape))
        }

        output = model(dummy_input)

        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(output.shape, (1, 128, 128, 64, 1))
if __name__ == '__main__':
    unittest.main()