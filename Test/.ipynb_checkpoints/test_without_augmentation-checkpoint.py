__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import unittest
import tensorflow as tf
from tensorflow.keras import layers
from unittest.mock import patch
from pyMEAL.without_augmentation import create_encoder, refined_diffusion_model

class TestEncoderBlockNA(unittest.TestCase):
    def test_encoder_output_shape(self):
        input_tensor = tf.random.uniform((1, 128, 128, 64, 1))
        with patch("pyMEAL.without_augmentation.bcs.refined_residual_block", side_effect=lambda x, f: x):
            output = create_encoder(input_tensor)
        self.assertEqual(output.shape, (1, 32, 32, 16, 128)) 
if __name__ == "__main__":
    unittest.main()

class TestRefinedDiffusionModelNA(unittest.TestCase):
    def setUp(self):
        self.input_shape = (128, 128, 64, 1)

    def test_model_output_shape(self):
        model = refined_diffusion_model(self.input_shape)
        dummy_input = tf.random.uniform((1, *self.input_shape))
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 128, 128, 64, 1))

    def test_model_structure(self):
        model = refined_diffusion_model(self.input_shape)
        self.assertTrue(any(isinstance(layer, layers.UpSampling3D) for layer in model.layers))
        self.assertTrue(any(isinstance(layer, layers.Conv3DTranspose) for layer in model.layers))
        self.assertEqual(model.output_shape, (None, 128, 128, 64, 1))
if __name__ == "__main__":
    unittest.main()

class TestIntegrationEncoderDecoderNA(unittest.TestCase):
    def test_end_to_end_pass(self):
        model = refined_diffusion_model((128, 128, 64, 1))
        input_tensor = tf.random.normal((1, 128, 128, 64, 1))
        output = model(input_tensor)
        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(output.shape, (1, 128, 128, 64, 1))
if __name__ == "__main__":
    unittest.main()


