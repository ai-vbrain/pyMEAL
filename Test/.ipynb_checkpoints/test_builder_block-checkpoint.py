__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import unittest
import pyMEAL.builder_block as BD
import tensorflow as tf
import matplotlib as plt
from unittest.mock import patch


# Constants
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 1

# Real image paths
ct_path = './CTScan data/processed_data/proceed_CT/te/sub-OAS30001_sess-d3132_CT.nii.gz'
t1_path = './CTScan data/processed_data/proceed_T1/te/sub-OAS30001_ses-d3132_T1w_be.nii.gz'

class TestAugmentationLayersBD(unittest.TestCase):

    def setUp(self):
        self.sample_input = tf.random.uniform(shape=(1, *INPUT_SHAPE))

    def test_flip_augmentation(self):
        layer = BD.FlipAugmentation()
        output = layer(self.sample_input)
        self.assertEqual(output.shape, self.sample_input.shape)

    def test_rotate_augmentation(self):
        layer = BD.RotateAugmentation()
        output = layer(self.sample_input)
        self.assertEqual(output.shape, self.sample_input.shape)

    def test_crop_augmentation(self):
        crop_size = (64, 64, INPUT_SHAPE[2], 1)
        layer = BD.CropAugmentation(crop_size, INPUT_SHAPE)
        output = layer(self.sample_input)
        self.assertEqual(output.shape, self.sample_input.shape)

    def test_intensity_augmentation(self):
        layer = BD.IntensityAugmentation()
        output = layer(self.sample_input)
        self.assertEqual(output.shape, self.sample_input.shape)
if __name__ == "__main__":
    unittest.main()

class TestModelArchitecturesBD(unittest.TestCase):

    def test_refined_diffusion_model_output(self):
        model = BD.refined_diffusion_model()
        sample_input = tf.random.uniform((1, *INPUT_SHAPE))
        output = model(sample_input)
        self.assertEqual(len(output.shape), 5)

    def test_controller_block(self):
        dummy_features = tf.random.uniform(shape=(1, 4, 64))
        controller = BD.build_controller_block(dummy_features)
        output = controller(dummy_features)
        self.assertEqual(output.shape, dummy_features.shape)

    def test_build_multistream_model_output(self):
        model = BD.build_multistream_model()
        sample_input = tf.random.uniform((1, *INPUT_SHAPE))
        output = model(sample_input)
        self.assertEqual(output.shape, sample_input.shape)
if __name__ == "__main__":
    unittest.main()
