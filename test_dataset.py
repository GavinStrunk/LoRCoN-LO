import unittest

import numpy as np

from kitti_odometry import KittiDataset
import utils.process_data as process_data


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # self.custom = KittiDataset
        preprocessed_folder = "/home/gstrunk/Projects/1762-AUTOMATIC/Software/automatic/third-party/LoRCoN-LO/data/KITTI/preprocessed_data"
        relative_pose_folder = "/home/gstrunk/Projects/1762-AUTOMATIC/Software/automatic/third-party/LoRCoN-LO/data/KITTI/relative_pose"
        data_seqs = ["01"]
        seq_sizes = {}
        rnn_size = 4
        image_height = 64
        image_width = 900
        dni_size = 5
        normal_size = 3
        depth_name = "depth"
        intensity_name = "intensity"
        normal_name = "normal"

        seq_sizes = process_data.count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
        Y_data = process_data.process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes)
        test_idx = np.arange(0, seq_sizes[data_seqs[0]] - (rnn_size), dtype=int)
        self.true_data = process_data.LoRCoNLODataset(preprocessed_folder, Y_data, test_idx, seq_sizes, rnn_size,
                                                      image_width,
                                                      image_height, depth_name, intensity_name, normal_name, dni_size,
                                                      normal_size)

        self.custom = KittiDataset(
            '/home/gstrunk/Projects/1762-AUTOMATIC/Software/automatic/src/automatic/datasets/KITTI/odometry/dataset',
            input_data=['lidar_lorcon_lo'],
            kitti_sequences=data_seqs[0],
            output_data='rel_pose',
            data_sequence_size=4)

    def test_setup(self):
        self.assertEqual(True, True)  # add assertion here

    def test_data_shape(self):
        self.assertEqual(self.true_data[0][0].shape, self.custom[0][0].shape)

    def test_data_content(self):
        inputs, label = self.true_data[0]
        np.testing.assert_array_almost_equal(inputs, self.custom[0][0])

    def test_data_output_shape(self):
        inputs, label = self.true_data[0]
        self.assertEqual(label[-1, :].shape, self.custom[0][1].shape)

    def test_data_output_content(self):
        inputs, label = self.true_data[0]
        np.testing.assert_array_almost_equal(label[-1, :].numpy(), self.custom[0][1])

    def test_length(self):
        self.assertEqual(len(self.true_data), len(self.custom))


if __name__ == '__main__':
    unittest.main()
