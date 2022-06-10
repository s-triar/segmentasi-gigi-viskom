from vedo import *


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

#  AE AP BA CPL FAW
persons = ['AE', 'AP', 'BA', 'CPL','FAW']
# GSF HC KEC MJK MNF MRAI NR NT RK SA Sam SN. SS UR WPP YS ZR
# persons = ['GSF', 'HC', 'KEC', 'MJK', 'MNF'] # 'NR', 'NT', 'RK', 'SA', 'Sam', 'SN', 'SS', 'UR', 'WPP', 'YS', 'ZR'
# persons = ['MRAI' ,'NR', 'NT','RK', 'SA' ]
# persons = ['Sam', 'SN', 'SS', 'UR', 'WPP',]
# persons = [   'YS', 'ZR']
rahangs = ['Upper','Lower']
model_plot = Plotter(axes=1,N=len(persons)*2,interactive=True)
for ir in range(len(rahangs)):
    for ip in range(len(persons)):
        model_file = '{} {}JawScan_d_predicted_refined'.format(persons[ip], rahangs[ir])
        isContinue = False
        
        path= './out-train/{}.vtp'.format(model_file)
        if isContinue == False:
            path= './out-pre-trained/{}.vtp'.format(model_file)
            
        path= './ground-truth/{}.vtp'.format(model_file) # untuk melihat ground truth
        
        model=load(path)
        # print(model)
        model.celldata['mycolors']=np.zeros((len(model.celldata['Label']), 3),dtype=np.uint8)
        for i in range(len(model.celldata['Label'])):
            model.celldata['mycolors'][i]=map_label_color(model.celldata['Label'][i])

        model.celldata.select('mycolors')
        i_render=((len(persons)-1)*ir)+(ip+(1*ir))
        print(i_render)
        t = model_file.split('JawScan_d_predicted_refined')
        model_title = Text2D(t[0] + (' Continuous' if isContinue else ''), pos="bottom-middle")
        model_plot.add(model_title ,at=i_render)
        title = Text2D(persons[ip])
        model_plot.add(model,at=i_render)
        model_plot.add(title,at=i_render)

model_plot.show(at=None)