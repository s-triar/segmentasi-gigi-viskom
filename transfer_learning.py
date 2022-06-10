from operator import mod
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


model_path = './models/'
model_name = 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip'
model_altered_name = 'Mesh_Segementation_MeshSegNet_Max_15_classes_80samples_2uniques_ku_tf.tar'
num_classes_original = 15 #12 teeth + 1 gingiva + 2 wisdom teeth
num_channels_original = 15 #number of features
    
num_classes_target = 17 #14 teeth + 1 gingiva + 2 wisdom teeth
num_channels_target = 15 #number of features

model = MeshSegNet(num_classes=num_classes_original, num_channels=num_channels_original, with_dropout=True, dropout_p=0.5).to('cpu', dtype=torch.float)
opt = optim.Adam(model.parameters(), amsgrad=True)
checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
opt.load_state_dict(checkpoint['optimizer_state_dict'])
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

if(num_channels_original!=num_channels_target):
    out_channel_in = model.mlp1_conv1.out_channels
    model.mlp1_conv1 = torch.nn.Conv1d(num_channels_target, out_channel_in, 1)
if(num_classes_original!=num_classes_target):
    in_channel_out = model.output_conv.in_channels
    model.output_conv = torch.nn.Conv1d(in_channel_out, num_classes_target, 1)

torch.save({'epoch': epoch_init+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    os.path.join(model_path, model_altered_name))

print(model.output_conv.weight)

print(model.output_conv.weight.shape)
print(opt)