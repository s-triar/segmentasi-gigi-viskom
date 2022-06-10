import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import csv

if __name__ == '__main__':

    data_path = './augmentation_vtk_data/'
    output_path = './'
    num_augmentations = 20
    train_size = 0.8
    archs=['upper','lower']
    for arch in archs:
        
        sample_ku=[]
        for i in glob.glob(data_path+"*.vtp"):
            if(arch in i):
                sample_ku.append(i)
        sample_ku = np.asarray(sample_ku)
        i_cv = 0
        kf = KFold(n_splits=6, shuffle=True, random_state=234)
        for train_idx, test_idx in kf.split(sample_ku):
            print(train_idx, test_idx)
            i_cv += 1
            print('Round:', i_cv)

            train_list, test_list = sample_ku[train_idx], sample_ku[test_idx]
            print(train_list, test_list)
            train_list, val_list = train_test_split(train_list, train_size=0.8, shuffle=True)

            print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)
            
            # training
            with open(os.path.join(output_path, './list_input/{0}_train_list_{1}.csv'.format(arch, i_cv)), 'w', newline='',encoding='utf-8' ) as file:
                writer = csv.writer(file)
                for f in train_list:
                    writer.writerow([f])
                    
            # validation
            with open(os.path.join(output_path, './list_input/{0}_val_list_{1}.csv'.format(arch, i_cv)), 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for f in val_list:
                    writer.writerow([f])
                    
                
            #test
            test_df = pd.DataFrame(data=test_list, columns=['Test ID'])
            test_df.to_csv('./list_input/{0}_test_list_{1}.csv'.format(arch, i_cv), index=False, encoding='utf-8')
            
