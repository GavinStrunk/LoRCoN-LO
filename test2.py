import os
import yaml
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim

import utils.models
from kitti_odometry import KittiDataset

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import model as model_util
import plot
import process_data
import common

if __name__ == "__main__":
    config_filename = 'config/config.yml'
    config = yaml.load(open(config_filename), yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"]=config["cuda_visible_devices"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    cuda = torch.device('cuda')
    seq_sizes= {}
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    torch.set_num_threads(num_workers)

    preprocessed_folder = config["preprocessed_folder"]
    pose_folder = config["pose_folder"]
    relative_pose_folder = config["relative_pose_folder"]

    data_seqs = config["data_seqs"].split(",")
    test_seqs = config["test_seqs"].split(",")

    rnn_size = config["rnn_size"]
    image_width = config["image_width"]
    image_height = config["image_height"]

    depth_name = config["depth_name"]
    intensity_name = config["intensity_name"]
    normal_name = config["normal_name"]
    dni_size = config["dni_size"]
    normal_size = config["normal_size"]

    cp_folder = config["cp_folder"]

    dataset = config["dataset"]

    checkpoint_path = os.path.join(cp_folder, config["checkpoint_path"])

    seq_sizes = process_data.count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)
    Y_data = process_data.process_input_data(preprocessed_folder, relative_pose_folder, data_seqs, seq_sizes)
    
    start_idx = 0
    end_idx = 0
    # train_idx = np.array([], dtype=int)
    test_idx = np.array([], dtype=int)
    for seq in data_seqs:
        end_idx += seq_sizes[seq] - 1
        # if seq in test_seqs:
        test_idx = np.append(test_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
        # else:
        #     train_idx = np.append(train_idx, np.arange(start_idx, end_idx - (rnn_size - 1), dtype=int))
        start_idx += seq_sizes[seq] - 1
    
    test_data = process_data.LoRCoNLODataset(preprocessed_folder, Y_data, test_idx, seq_sizes, rnn_size, image_width, image_height, depth_name, intensity_name, normal_name, dni_size, normal_size)
    # test_data = KittiDataset('/home/gstrunk/Projects/1762-AUTOMATIC/Software/automatic/src/automatic/datasets/KITTI/odometry/dataset',
    #                        input_data=['lidar_lorcon_lo'],
    #                        kitti_sequences="05",
    #                        output_data='rel_pose',
    #                        data_sequence_size=4)
    
    test_dataloader = DataLoader(test_data, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    model = utils.models.load_lorconlo(checkpoint_path)
    # model = model_util.LoRCoNLO(batch_size=batch_size, batchNorm=False, device=device)

    # criterion = model_util.WeightedLoss(learn_hyper_params=False)
    # optimizer = optim.Adagrad(model.parameters(), lr=0.0005)

    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Save model using torchscript
    # model.eval()
    # x, labels = next(iter(test_dataloader))
    # x = Variable(x.float().to(device))
    # with torch.no_grad():
    #     print(model(x).shape)
    #     traced_cell = torch.jit.trace(model, (x))
    # torch.jit.save(traced_cell, "lorconlo.pth")
    #
    # # Load model from torchscript
    # model2 = torch.jit.load("lorconlo.pth")

    Y_estimated_data = np.empty((0, 6), dtype=np.float64)
    test_data_loader_len = len(test_dataloader)
    test_loss = 0.0

    # inputs, labels = test_data[0]
    # testd = Variable(inputs.float().to(device))
    # testd = testd[None, :]
    # print(testd.shape)
    # outputs = model(testd)
    #
    # import json
    # with open('../../../Projects/1762-AUTOMATIC/Software/automatic/src/automatic/test.txt', 'w') as file:
    #     file.write(json.dumps({'inputs': testd.cpu().detach().numpy().tolist(), 'labels': labels.numpy().tolist(), 'outputs': outputs.cpu().detach().numpy().tolist()}))

    with torch.no_grad():
        for idx, data in tqdm(enumerate(test_dataloader)):
        # for idx, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs, labels = Variable(inputs.float().to(device)), Variable(labels.float().to(device))
            outputs = model(inputs)
            # print(outputs.shape)
            Y_estimated_data = np.vstack((Y_estimated_data, outputs[:,-1,:].cpu().numpy()))
            # print(Y_estimated_data.shape)
            test_loss += model_util.RMSEError(outputs, labels).item()
    print(f"Test loss is {test_loss / test_data_loader_len}")

    Y_origin_data = common.get_original_poses(pose_folder, preprocessed_folder, data_seqs)
    seq_sizes = {}
    seq_sizes = process_data.count_seq_sizes(preprocessed_folder, data_seqs, seq_sizes)

    plot.plot_gt(Y_origin_data, pose_folder, preprocessed_folder, data_seqs, seq_sizes, dataset=dataset, dim="3d")
    plot.plot_results(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset=dataset, dim="3d")
    common.save_poses(Y_origin_data, Y_estimated_data, data_seqs, rnn_size, seq_sizes, dataset=dataset)