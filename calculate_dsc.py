import glob
import os
from losses_and_metrics_for_mesh import *
import torch.nn as nn
import torch
from vedo import *
import numpy as np

def get_label(mesh):
    patch_size=9999
    labels = mesh.getCellArray('Label').astype('int32').reshape(-1, 1)
    Y = labels
    YT = np.zeros([patch_size,1], dtype='int32')
    positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
    negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

    num_positive = len(positive_idx) # number of selected tooth cells

    if num_positive > patch_size: # all positive_idx in this patch
        positive_selected_idx = np.random.choice(positive_idx, size=patch_size, replace=False)
        selected_idx = positive_selected_idx
    else:   # patch contains all positive_idx and some negative_idx
        num_negative = patch_size - num_positive # number of selected gingiva cells
        positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
        negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
        selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

    selected_idx = np.sort(selected_idx, axis=None)
    YT[:] = Y[selected_idx, :]
    
    YT = YT.transpose(1, 0)
    return torch.from_numpy(YT)

archs=['Upper','Lower']
for arch in archs:
    ground_filenames=[]
    for i in glob.glob('./src/*.*'):
        a = i.split('\\')[-1]
        if(arch.lower() in a):
            ground_filenames.append(a)
    predicted_filenames=[]
    for i in glob.glob('./src-temp/*.*'):
        a = i.split('\\')[-1]
        if(arch in a):
            predicted_filenames.append(a)
    for i in range(5):
        gt = ground_filenames[i]
        pr = predicted_filenames[i]
        
        data_gt = load("./src/"+gt)
        data_pr = load("./src-temp/"+pr)

        aa = get_label(data_gt)
        bb = get_label(data_pr)
        class_weights = torch.ones(15).to('cpu', dtype=torch.float)
        one_hot_labels_aa = nn.functional.one_hot(aa.long()[:], num_classes=15)
        one_hot_labels_bb = nn.functional.one_hot(bb.long()[:], num_classes=15)
        dsc = weighting_DSC(one_hot_labels_aa, one_hot_labels_bb, class_weights)
        print(pr,"\t",dsc.data)