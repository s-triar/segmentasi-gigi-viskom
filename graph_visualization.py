import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset import *
from meshsegnet import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd

if __name__ == '__main__':
    previous_check_point_path = './models'
    previous_check_point_name = 'Mesh_Segementation_MeshSegNet_Max_15_classes_80samples_2uniques_ku.tar'
    
    model_path = './models/'
    # model_name = 'Mesh_Segementation_MeshSegNet_15_classes_72samples' #remember to include the project title (e.g., ALV)
    model_name = 'Mesh_Segementation_MeshSegNet_Max_15_classes_80samples_2uniques_ku' #remember to include the project title (e.g., ALV)
    checkpoint_name = 'Mesh_Segementation_MeshSegNet_Max_15_classes_80samples_2uniques_ku'
    num_classes = 17 #14 teeth + 1 gingiva + 2 wisdom teeth
    num_channels = 15 #number of features
    
    # set plotter
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=model_name)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    checkpoint = torch.load(os.path.join(previous_check_point_path, previous_check_point_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    # opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_init = checkpoint['epoch']
    losses = checkpoint['losses']
    mdsc = checkpoint['mdsc']
    msen = checkpoint['msen']
    mppv = checkpoint['mppv']
    val_losses = checkpoint['val_losses']
    val_mdsc = checkpoint['val_mdsc']
    val_msen = checkpoint['val_msen']
    val_mppv = checkpoint['val_mppv']
    del checkpoint
    for i_epoch in range(len(losses)):
        plotter.plot('loss', 'train', 'Loss', i_epoch+1, losses[i_epoch])
        plotter.plot('DSC', 'train', 'DSC', i_epoch+1, mdsc[i_epoch])
        plotter.plot('SEN', 'train', 'SEN', i_epoch+1, msen[i_epoch])
        plotter.plot('PPV', 'train', 'PPV', i_epoch+1, mppv[i_epoch])
        plotter.plot('loss', 'val', 'Loss', i_epoch+1, val_losses[i_epoch])
        plotter.plot('DSC', 'val', 'DSC', i_epoch+1, val_mdsc[i_epoch])
        plotter.plot('SEN', 'val', 'SEN', i_epoch+1, val_msen[i_epoch])
        plotter.plot('PPV', 'val', 'PPV', i_epoch+1, val_mppv[i_epoch])