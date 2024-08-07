from enum import Enum
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tvt
from typing import Tuple, Any, Optional, Callable
import pykitti
import scipy.spatial.transform as tf


class AutomaticDataset(Dataset):
    """
    base class/interface for all datasets used for AUTOMATIC
    any code that is common to both KITTI and MVSEC goes here!
    """

    def __init__(self, base_filename, input_data=['camera', 'lidar', 'imu'], output_data='pose') -> None:
        """
        base_filename: 
        input_data: 
        output_data: 
        """
        self.input_data = input_data
        self.output_data = output_data
        self.filename = base_filename

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        pass


class KittiDataset(AutomaticDataset):
    """
    the dataset from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
    """

    def __init__(self, 
                 base_filename, 
                 kitti_sequences="00,01,02,04,05,06,07,08,09,10", 
                 input_data=['camera', 'lidar_lorcon_lo', 'imu'], 
                 transform= tvt.Compose([tvt.PILToTensor()]),
                 output_data='pose',
                 data_sequence_size=1
                 ) -> None:
        """
        Parameters
        ----------
        base_filename : str
            the location of the Kitti dataset
        kitti_sequences : str
            comma separated numbers (02d) that define the sequences used in the KITTI dataset 
        input_data : list(str)
            The types of sensor data the dataset should provide. Some data types are raw, while otheres are preprocessed
        output_data: str
            the type of output/ ground truth data the dataset should provide. Examples include absolute 'pose' or relative 'rel_pose'
        data_sequence_size: int
            training recurrent neural networks requires sending consecutive sensor data. this specifies 
            the number consecutive sensor datum points to stack

        """

        super().__init__(base_filename,input_data,output_data)

        self.data_sequence_size = data_sequence_size
        self.kitti_sequences = kitti_sequences.split(',')
        self.kitti_sequences_length = dict() #the number of data samples in a given length
        self.dataset = dict()
        self.index_start = dict()  #

        idx = 0
        for seq in self.kitti_sequences:
            self.dataset[seq] = pykitti.odometry(base_filename,seq)
            self.kitti_sequences_length[seq] = len(self.dataset[seq]) - self.data_sequence_size
            self.index_start[seq] = idx
            idx = idx + self.kitti_sequences_length[seq]
        self.length = idx

        self.img_transform = transform

        self.repo_dir = os.environ.get('REPO_DIR_ROOT')

        for device in input_data:
            try:
                if device == 'camera':
                    # this data is loaded directly using pykitti.odometry
                    pass
                if device == 'camera_consecutive':
                    # this data is loaded directly using pykitti.odometry
                    pass
                elif device == 'lidar':
                    pass
                elif device == 'lidar_lorcon_lo':
                    self.lidar_lorcon_lo_path = os.path.join(self.repo_dir, 'KITTI/odometry/dataset/sequences/%s/preprocessed_lorcon_lo_data/')
                elif device == 'imu':
                    self.imu_path = os.path.join(self.repo_dir, 'KITTI/odometry/dataset/sequences/%s/oxts/')
            except:
                raise ImportError(f'no {device} found in {self.filename}')
            
        if self.output_data == "pose":
            # now ingest GPS data as ground truth
            self.poses = dict()
            for seq in self.kitti_sequences:
                Nposes = len(self.dataset[seq].poses)
                print('number of poses: %d' % Nposes)
                self.poses[seq] = np.nan * np.ones([6, Nposes])
                for i in np.arange(Nposes):
                    se3 = self.dataset[seq].poses[i]  # each pose is an SE(3) homogeneous transformation matrix
                    rot = tf.Rotation.from_matrix(se3[0:3, 0:3])
                    euler = rot.as_euler(
                        'ZYX')  # RT 3003 uses intrinsic Z-Y-X angles note that "xyz" and "ZYX" result in the same element SO(3)
                    # order = heading, pitch, roll
                    euler = np.array([euler[2], euler[1], euler[0]])  # rearrange to roll, pitch yaw
                    p = se3[0:3, 3]
                    p = np.concatenate((p, euler), axis=0)
                    self.poses[seq][:, i] = p
        elif self.output_data == "rel_pose":
            self.rel_poses = dict()
            for seq in self.kitti_sequences:
                rel_pose_file = os.path.join(self.repo_dir, 'KITTI/odometry/dataset/relative_poses/%s.txt' % seq)
                self.rel_poses[seq] = np.loadtxt(rel_pose_file)
        else:
            raise ImportError(f'{self.output_data} not a valid form of output!')

        # for N samples
        # self.input = [time, ]
        # self.output = torch.tensor(pose)

    def index_to_seq_index(self,index):
        """
        calculate sequence number and sequence index from index      
        """
        for seq in self.kitti_sequences:
            if (index >= self.index_start[seq]):
                sequence = seq
                i = index - self.index_start[seq]
        return (sequence,i)
    
    def index_to_seq_index_iterator(self,index,sequence_length):
        """
        calculate sequence number and index iterator of length sequence_length from index 
        """
        for seq in self.kitti_sequences:
            if (index >= self.index_start[seq]):
                sequence = seq
                i = index - self.index_start[seq]
        index_iterator = range(i,(i+sequence_length))
        
        #(self.dataset[seq](i+sequence_length-1) #assert that this exists?
        return (sequence,index_iterator)
    
    def prev_seq_index(self,index):
        """
        gets the previous sequence/index pair from a given index.
        useful if you want consecutive sensor measurements combined, but preserving discontinuities in the data sequence.
        """
        seq, i = self.index_to_seq_index(index)

        if i == 0:
            # at the start of the sequence
            return (seq, i)
        else:
            return (seq,i-1)
        

    def prev_seq_index(self,seq,i):
        """
        gets the previous sequence/index pair from a given index.
        useful if you want consecutive sensor measurements combined, but preserving discontinuities in the data sequence.
        """
        if i == 0:
            #at the start of the sequence
            return (seq,i)
        else:
            return (seq,i-1)
        
    def next_seq_index(self,index):
        """
        gets the next sequence/index pair from a given index.
        useful if you want consecutive sensor measurements combined, but preserving discontinuities in the data sequence.
        """
        seq, i = self.index_to_seq_index(index)

        if i == self.kitti_sequences_length[seq]-1:
            #at the end of the sequence
            return (seq,i)
        else:
            return (seq, i + 1)

    def __len__(self) -> int:
        return self.length

    # def __getitem__(self, index) -> [Tuple[Any, Any]]:
    #     seq, indices = self.index_to_seq_index_iterator(index, self.data_sequence_size)
    #
    #     input_data = list()
    #     for device in self.input_data:
    #         if device == 'camera':
    #             data_to_stack = [self.img_transform(self.dataset[seq].get_cam3(idx)) for idx in indices]
    #             input_data.append(torch.stack(data_to_stack))
    #         elif device == 'camera_consecutive':
    #             raise ImportError(f'{device} not supported. Did you mean camera?')
    #         elif device == 'lidar_lorcon_lo':
    #             data_to_stack = []
    #             for idx in indices:
    #                 # Use previously loaded lidar data
    #                 filename = "{:06d}.npy".format(idx)
    #                 depth, intensity, normal = self.get_lidar(filename, seq)
    #                 current_data = torch.cat([depth, intensity, normal], dim=0)
    #
    #                 # For previous lidar data
    #                 _, idx_prev = self.prev_seq_index(seq, idx)
    #                 filename_prev = "{:06d}.npy".format(idx_prev)
    #                 depth_prev, intensity_prev, normal_prev = self.get_lidar(filename_prev, seq)
    #                 prev_data = torch.cat([depth_prev, intensity_prev, normal_prev], dim=0)
    #
    #                 combined_data = torch.cat([prev_data, current_data], dim=0)
    #                 data_to_stack.append(combined_data)
    #             input_data.append(torch.stack(data_to_stack))
    #         elif device == 'imu':
    #             data_to_stack = []
    #             for idx in indices:
    #                 filename = "{:010d}.txt".format(idx)
    #                 oxts_data = np.loadtxt(os.path.join(self.imu_path % seq, filename))
    #                 imu = torch.tensor(self.oxts_to_imu(oxts_data))
    #                 data_to_stack.append(imu)
    #             input_data.append(torch.stack(data_to_stack))
    #         else:
    #             raise ImportError(f'no {device} found in {self.filename}')
    #
    #     if len(input_data) == 1:
    #         input_data = input_data[0]
    #
    #     _, last = indices[-1], indices[-1]
    #     if self.output_data == "pose":
    #         output = self.poses[seq][:, last]
    #     elif self.output_data == "rel_pose":
    #         output = self.rel_poses[seq][last, :]
    #     else:
    #         raise ImportError(f'{self.output_data} not a valid form of output!')
    #
    #     return (input_data, output)

    def __getitem__(self, index) -> [Tuple[Any, Any]]:
        # Loads and returns a sample from the dataset at a given index.

        seq, indices = self.index_to_seq_index_iterator(index,self.data_sequence_size)

        input_data = list()

        for device in self.input_data:
            try:
                if device == 'camera':
                    #we've only downloaded cameras 2 and 3, that's the color data
                    #deepVO only uses the right color camera (camera 3)
                    data_to_stack = []
                    for idx in indices:
                        cam3 = self.dataset[seq].get_cam3(idx)
                        data_to_stack.append(self.img_transform(cam3))
                    input_data.append(torch.stack(data_to_stack))
                if device == 'camera_consecutive':
                    raise  ImportError(f'{device} not supported. Did you mean camera?')

                    #tack on the previous image to the current image
                    seq_prev,idx_prev = self.prev_seq_index(index)
                    cam3 = self.dataset[seq].get_cam3(idx)
                    cam3_prev = self.dataset[seq_prev].get_cam3(idx_prev)
                    cam3t = (self.img_transform(cam3))
                    cam3t_prev = (self.img_transform(cam3_prev))
                    combined = torch.cat([cam3t_prev,cam3t],dim=0)
                    input_data.append(combined)
                elif device == 'lidar':
                    raise ImportError(f'{device} not supported. Did you mean lidar_lorcon_lo?')
                elif device == 'lidar_lorcon_lo':
                    data_to_stack = []
                    for idx in indices:
                        #grab the previous Lidar
                        # _,idx_prev = self.prev_seq_index(seq,idx)
                        filename = "{:06d}".format(idx) + ".npy"
                        depth,intensity,normal = self.get_lidar(filename, seq)
                        data = torch.cat([depth,intensity,normal],dim=0)
                        #data_to_stack.append(data)

                        #now grab current lidar
                        filename = "{:06d}".format(idx+1) + ".npy"
                        depth,intensity,normal = self.get_lidar(filename, seq)
                        data = torch.cat([data,depth,intensity,normal], dim=0)
                        data_to_stack.append(data)
                    input_data.append(torch.stack(data_to_stack))
                elif device == 'imu':
                    data_to_stack = []
                    for idx in indices:
                        filename = "{:010d}".format(idx) + ".txt"
                        oxts_data = np.loadtxt(os.path.join(self.imu_path % seq, filename))
                        imu = torch.tensor(self.oxts_to_imu(oxts_data))
                        data_to_stack.append(imu)

                    input_data.append(torch.stack(data_to_stack))
            except:
                raise ImportError(f'no {device} found in {self.filename}')

        #for a single input data, the list should be collapsed into a single element because it's training a deep NN
        if len(input_data)==1:
            input_data = input_data[0]


        #note that for output data, we grab the last element in the iterator and only return the final output
        *_, last = indices

        if self.output_data == "pose":
            output = self.poses[seq][:,last]
        elif self.output_data == "rel_pose":
           output = self.rel_poses[seq][last,:]
        else:
            raise ImportError(f'{self.output_data} not a valid form of output!')

        return (input_data, output)

    def get_lidar(self,filename,seq):
        depth = torch.tensor(np.load((self.lidar_lorcon_lo_path % seq) + 'depth/' + filename))
        depth = depth[None, :]
        depth /= 255.0

        intensity = torch.tensor(np.load(self.lidar_lorcon_lo_path % seq + '/intensity/' + filename))
        intensity = intensity[None, :]
        intensity /= 255.0

        normal = torch.tensor(np.load(self.lidar_lorcon_lo_path % seq + '/normal/' + filename))
        normal = torch.permute(normal,(2,0,1))
        normal = (normal + 1.0) / 2.0

        return (depth, intensity, normal)

    def oxts_to_imu(self,oxts_data):
        """
        converts a line read from oxts_data as numpy to the desired IMU output
        grabs the following:
        ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
        ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
        az:    acceleration in z, i.e. in direction of vehicle top (m/s^2)

        """
        return np.array([oxts_data[11], oxts_data[12], oxts_data[13]])


"""
on recurrent layers
https://pytorch.org/docs/stable/nn.html#recurrent-layers
"""

"""
DeepVO using pytorch https://github.com/ChiWeiHsiao/DeepVO-pytorch/blob/master/Dataloader_loss.py
sequential data is just a list of stuff, e.g. (features,ground_truth)
"""

if __name__ == "__main__":
    repo_dir = os.environ.get('REPO_DIR_ROOT')
    if repo_dir is None:
        raise EnvironmentError(
            "Could not find environment variable REPO_DIR_ROOT. Set this variable to the root directory of the dataset folder")
    print(repo_dir)
    basepath = repo_dir + '/KITTI/odometry/dataset'
    print(basepath)
    dataset = KittiDataset(basepath, output_data="rel_pose")

    dataset[1]
    dataset[10]
    dataset[100]
    dataset[1000]
    dataset[10000]
