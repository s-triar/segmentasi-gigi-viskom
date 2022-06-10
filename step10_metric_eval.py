from numpy import mean
from vedo import *
import glob
import os
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, precision_recall_fscore_support, jaccard_score
import pandas as pd

persons = ['AE', 'AP', 'BA', 'CPL','FAW']
rahangs = ['Upper','Lower']
akurasi =[]
f1=[]
ff=[]
name = []
ppv=[]
sen=[]
akurasi_c =[]
f1_c=[]
ff_c=[]
ppv_c=[]
sen_c=[]
for r in rahangs:
    for p in persons:
        model_file = '{} {}JawScan_d_predicted_refined'.format(p,r)

        path_truth= './ground-truth/{}.vtp'.format(model_file)
        path_pred = './out-pre-trained/{}.vtp'.format(model_file)

        model_plot = Plotter(axes=1,N=4,interactive=True)


        model_truth=load(path_truth)
        model_pred=load(path_pred)

        y_truth = model_truth.celldata['Label']
        y_pred = model_pred.celldata['Label']

        labels =[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        a = accuracy_score(y_true=y_truth,y_pred=y_pred, normalize=True)
        f = f1_score(y_true=y_truth,y_pred=y_pred,labels=labels, average='macro')
        name.append(model_file)
        akurasi.append(a)
        f1.append(f)
        prf = precision_recall_fscore_support(y_true=y_truth,y_pred=y_pred,labels=labels, average='macro', beta=1)
        ff.append(prf[2])
        ppv.append(prf[0])
        sen.append(prf[1])
        
for r in rahangs:
    for p in persons:
        model_file = '{} {}JawScan_d_predicted_refined'.format(p,r)

        path_truth= './ground-truth/{}.vtp'.format(model_file)
        path_pred = './out-train/{}.vtp'.format(model_file)

        model_plot = Plotter(axes=1,N=4,interactive=True)


        model_truth=load(path_truth)
        model_pred=load(path_pred)

        y_truth = model_truth.celldata['Label']
        y_pred = model_pred.celldata['Label']

        labels =[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
        a = accuracy_score(y_true=y_truth,y_pred=y_pred, normalize=True)
        f = f1_score(y_true=y_truth,y_pred=y_pred, average='macro')
        akurasi_c.append(a)
        f1_c.append(f)
        prf = precision_recall_fscore_support(y_true=y_truth,y_pred=y_pred, average='macro', beta=1)
        ff_c.append(prf[2])
        ppv_c.append(prf[0])
        sen_c.append(prf[1])
        
        
name.append("rata2")
akurasi.append(mean(akurasi))
ff.append(mean(ff))
ppv.append(mean(ppv))
sen.append(mean(sen))

akurasi_c.append(mean(akurasi_c))
ff_c.append(mean(ff_c))
ppv_c.append(mean(ppv_c))
sen_c.append(mean(sen_c))

df = pd.DataFrame(list(zip(name, akurasi,ff,ppv,sen, akurasi_c,ff_c,ppv_c,sen_c)), columns=['model','akurasi','f1 score','Prec','Recall', 'akurasi C','f1 score C','Prec C','Recall C'])
print(df)

