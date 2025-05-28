__authors__ = 'Adeleke Maradesa, and Abdulmojeed Ilyas'

__date__ = '24th May, 2025'


import unittest
import tensorflow as tf
import numpy as np
import nibabel as nib
# from unittest.mock import patch
from pyMEAL.encoder_concatenation import create_dataset, build_multistream_model
from unittest.mock import patch

# Constants
INPUT_SHAPE = (128, 128, 64, 1)
TARGET_SHAPE = INPUT_SHAPE[:3]
BATCH_SIZE = 2

# Real file paths for integration testing
ct_path = './CTScan data/processed_data/proceed_CT/te/sub-OAS30001_sess-d3132_CT.nii.gz'
t1_path = './CTScan data/processed_data/proceed_T1/te/sub-OAS30001_ses-d3132_T1w_be.nii.gz'

'''
Test Encoder Cocatenation (Encoder_concatenationpy)
'''

# -------------------- UNIT TESTS --------------------

class TestDatasetStructureCC(unittest.TestCase):
    @patch('pyMEAL.encoder_concatenation.bcs.create_dataset')
    def test_dataset_streams_structure(self, mock_create_dataset):
        # Mock a basic dataset of input/target volumes
        def gen():
            for _ in range(2):
                input_vol = tf.random.uniform(INPUT_SHAPE)
                target_vol = tf.random.uniform(TARGET_SHAPE + (1,))
                yield input_vol, target_vol

        mock_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(INPUT_SHAPE, tf.float32),
                tf.TensorSpec(TARGET_SHAPE + (1,), tf.float32)
            )
        )
        mock_create_dataset.return_value = mock_ds

        dataset = create_dataset("dummy_input", "dummy_target", batch_size=2)
        for inputs, target in dataset.take(1):
            self.assertEqual(len(inputs), 4)
            for stream in inputs:
                self.assertEqual(stream.shape, INPUT_SHAPE)
            self.assertEqual(target.shape, TARGET_SHAPE + (1,))
if __name__ == "__main__":
    unittest.main()


class TestMultistreamModelCC(unittest.TestCase):
    def test_model_structure_and_forward_pass(self):
        model = build_multistream_model(INPUT_SHAPE)
        self.assertEqual(len(model.inputs), 4)
        self.assertEqual(model.output_shape, (None,) + INPUT_SHAPE)

        fake_batch = [tf.random.uniform((BATCH_SIZE,) + INPUT_SHAPE) for _ in range(4)]
        pred = model(fake_batch)
        self.assertEqual(pred.shape, (BATCH_SIZE,) + INPUT_SHAPE)
if __name__ == "__main__":
    unittest.main()


# -------------------- INTEGRATION TEST --------------------

class Testbuild_multistream_modelCC(unittest.TestCase):
    def load_nii_as_tensor(self, path, target_shape):
        img = nib.load(path).get_fdata()
        if img.ndim == 3:
            img = img[..., np.newaxis]  # Ensure channel

        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

        # Resize x-y per slice, keeping z constant
        slices = tf.unstack(img_tensor, axis=2)
        resized_slices = [tf.image.resize(slice_, target_shape[:2]) for slice_ in slices]
        img_resized = tf.stack(resized_slices, axis=2)

        # Crop or pad depth to target
        current_depth = img_resized.shape[2]
        target_depth = target_shape[2]
        if current_depth < target_depth:
            pad = target_depth - current_depth
            img_resized = tf.pad(img_resized, [[0, 0], [0, 0], [0, pad], [0, 0]])
        elif current_depth > target_depth:
            img_resized = img_resized[:, :, :target_depth, :]

        return tf.reshape(img_resized, target_shape)

    def test_real_ct_input_through_model(self):
        ct_tensor = self.load_nii_as_tensor(ct_path, INPUT_SHAPE)
        ct_tensor = tf.expand_dims(ct_tensor, axis=0)  # Add batch dim

        model = build_multistream_model(INPUT_SHAPE)
        output = model([ct_tensor] * 4)
        self.assertEqual(output.shape, (1,) + INPUT_SHAPE)

if __name__ == "__main__":
    unittest.main()
