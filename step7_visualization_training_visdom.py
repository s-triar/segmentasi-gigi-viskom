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
    # models = ['MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip','MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip']
    archs = ['upper']
    folds = [1,2,3,4,56,]
    for a in archs:
        for fold in folds:
            print(a,fold)
    # for p_model in models:
            arch= a
            fld=fold
            id_k_fold=fld
            id_k_fold_before=fld
            model_path = './models/'
            model_name = 'cont_{0}_fold_{1}'.format(arch,str(id_k_fold)) #remember to include the project title (e.g., ALV)
            # model_name = p_model
            checkpoint_name = 'cont_{0}_fold_{1}'.format(arch,str(id_k_fold))
            previous_check_point_path = './models'
            previous_check_point_name = 'cont_{0}_fold_{1}.tar'.format(arch,str(id_k_fold_before))
            # previous_check_point_name = p_model
            num_classes = 15 #14 teeth + 1 gingiva
            num_channels = 15 #number of features
            num_epochs = 10
            num_workers = 0
            train_batch_size = 6 #12
            val_batch_size = 6 #12
            num_batches_to_print = 20
            

            # set plotter
            global plotter
            plotter = utils.VisdomLinePlotter(env_name=model_name)
            device = torch.device('cpu')
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
            

            # plot previous data
            for i_epoch in range(len(losses)):
                plotter.plot('loss', 'train', 'Loss', i_epoch+1, losses[i_epoch])
                plotter.plot('DSC', 'train', 'DSC', i_epoch+1, mdsc[i_epoch])
                plotter.plot('SEN', 'train', 'SEN', i_epoch+1, msen[i_epoch])
                plotter.plot('PPV', 'train', 'PPV', i_epoch+1, mppv[i_epoch])
                plotter.plot('loss', 'val', 'Loss', i_epoch+1, val_losses[i_epoch])
                plotter.plot('DSC', 'val', 'DSC', i_epoch+1, val_mdsc[i_epoch])
                plotter.plot('SEN', 'val', 'SEN', i_epoch+1, val_msen[i_epoch])
                plotter.plot('PPV', 'val', 'PPV', i_epoch+1, val_mppv[i_epoch])