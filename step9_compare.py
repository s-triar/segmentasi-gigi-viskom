from vedo import *
from sklearn.metrics import jaccard_score, f1_score

def map_label_color(label):
    if(label==0.0):
        return [220,220,220]
    elif(label==1.0):
        return [110,130,110]
    elif(label==2.0):
        return [100,130,160]
    elif(label==3.0):
        return [140,100,70]
    elif(label==4.0):
        return [80,130,90]
    elif(label==5.0):
        return [200,80,90]
    elif(label==6.0):
        return [100,210,90]
    elif(label==7.0):
        return [100,80,210]
    elif(label==8.0):
        return [110,180,190]
    elif(label==9.0):
        return [130,70,120]
    elif(label==10.0):
        return [180,200,170]
    elif(label==11.0):
        return [140,20,140]
    elif(label==12.0):
        return [130,190,0]
    elif(label==13.0):
        return [180,90,150]
    elif(label==14.0):
        return [0,160,160]



labels =[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
# persons =  AE AP BA CPL FAW
person = 'FAW'
arch = 'Upper' # Upper Lower
model_file = '{} {}JawScan_d_predicted_refined'.format(person, arch)
path_truth= './ground-truth/{}.vtp'.format(model_file)
path_pred = './out-pre-trained/{}.vtp'.format(model_file)
path_pred_c = './out-train/{}.vtp'.format(model_file)

model_plot = Plotter(axes=1,N=6,interactive=True)


model_truth=load(path_truth)
# model_pred=load(path_pred)

model_truth2 = load(path_truth)
model_truth2_c = load(path_truth)
model_pred2 = load(path_pred)
model_pred2_c = load(path_pred_c)


model_truth.celldata['mycolors']=np.zeros((len(model_truth.celldata['Label']), 3),dtype=np.uint8)
model_truth2.celldata['mycolors']=np.zeros((len(model_truth.celldata['Label']), 3),dtype=np.uint8)
model_pred2.celldata['mycolors']=np.zeros((len(model_truth.celldata['Label']), 3),dtype=np.uint8)
model_truth2_c.celldata['mycolors']=np.zeros((len(model_truth.celldata['Label']), 3),dtype=np.uint8)
model_pred2_c.celldata['mycolors']=np.zeros((len(model_truth.celldata['Label']), 3),dtype=np.uint8)

#compare labels
for i in range(len(model_truth.celldata['Label'])):
    model_truth.celldata['mycolors'][i]=map_label_color(model_truth.celldata['Label'][i])
    model_pred2.celldata['mycolors'][i]=map_label_color(model_pred2.celldata['Label'][i])
    model_pred2_c.celldata['mycolors'][i]=map_label_color(model_pred2_c.celldata['Label'][i])
    if(int(model_truth2.celldata['Label'][i])!=int(model_pred2.celldata['Label'][i])):
        model_truth2.celldata['mycolors'][i]=[0,0,0]
    else:
        model_truth2.celldata['mycolors'][i]=map_label_color(model_truth2.celldata['Label'][i])
    if(int(model_truth2_c.celldata['Label'][i])!=int(model_pred2_c.celldata['Label'][i])):
        model_truth2_c.celldata['mycolors'][i]=[0,0,0]
    else:
        model_truth2_c.celldata['mycolors'][i]=map_label_color(model_truth2_c.celldata['Label'][i])
    
model_truth.celldata.select('mycolors')
model_truth2.celldata.select('mycolors')
model_pred2.celldata.select('mycolors')
model_truth2_c.celldata.select('mycolors')
model_pred2_c.celldata.select('mycolors')

y_truth = model_truth.celldata['Label']
y_pred2 = model_pred2.celldata['Label']
y_pred2_c = model_pred2_c.celldata['Label']

iou = jaccard_score(y_true=y_truth,y_pred=y_pred2,labels=labels, average='macro')
iou_c = jaccard_score(y_true=y_truth,y_pred=y_pred2_c,labels=labels, average='macro')

f1 = f1_score(y_true=y_truth,y_pred=y_pred2,labels=labels, average='macro')
f1_c = f1_score(y_true=y_truth,y_pred=y_pred2_c,labels=labels, average='macro')

model_title = Text2D(person+' '+arch, pos="bottom-middle")
# model_plot.add(model_title ,at=0)
F1_PRED = Text2D('f1: {}'.format(str(f1)), pos="bottom-left")
IOU_PRED = Text2D('IOU: {}'.format(str(iou)), pos="bottom-right")
title_truth2 = Text2D("Intersection Pre-trained")
model_plot.add(model_truth2,at=4)
model_plot.add(title_truth2,at=4)
model_plot.add(F1_PRED,at=4)
model_plot.add(IOU_PRED,at=4)

title_truth = Text2D("Ground Truth")
title_pred2 = Text2D("Prediction")
model_plot.add(model_truth,at=3)
model_plot.add(title_truth,at=3)
model_plot.add(model_title,at=3)
model_plot.add(model_pred2,at=1)
model_plot.add(title_pred2,at=1)

F1_PRED_C = Text2D('f1: {}'.format(str(f1_c)), pos="bottom-left")
IOU_PRED_C = Text2D('IOU: {}'.format(str(iou_c)), pos="bottom-right")
title_truth2_c = Text2D("Intersection cont trained")
model_plot.add(model_truth2_c,at=5)
model_plot.add(title_truth2_c,at=5)
model_plot.add(F1_PRED_C,at=5)
model_plot.add(IOU_PRED_C,at=5)

title_pred2_c = Text2D("Pred cont trained")
model_plot.add(model_pred2_c,at=2)
model_plot.add(title_pred2_c,at=2)

model_plot.show(at=0)