from turtle import color
import pandas as pd
# from vedo import *
import glob
import os
from matplotlib import pyplot as plt
rahang = 'upper' # upper lower
path = './losses_metrics_vs_epoch_{}_fold_6.csv'.format(rahang) #mengambil fold yg terakhir saja. karena data fold sebelumnya juga tersimpan disana

df = []
for i in glob.glob(path):
    print(i)
    df_temp = pd.read_csv(i,)
    df_temp.rename(columns = {'Unnamed: 0':'Epoch'}, 
            inplace = True)
    if(len(df)==0):
        df = df_temp.copy()
    else:
        df = df.append(df_temp, ignore_index = True)

print(len(df))

print(df.index)
print(df.columns)


plt.figure(1)
plt.subplot(221)
plt.plot(df.index[-60:], df.iloc[-60:,1], label=df.columns[1])
plt.plot(df.index[-60:], df.iloc[-60:,5], label=df.columns[5])
plt.title(label=rahang)
plt.xlabel('Epoch')
plt.legend()
plt.subplot(222)
plt.plot(df.index[-60:], df.iloc[-60:,2], label=df.columns[2])
plt.plot(df.index[-60:], df.iloc[-60:,6], label=df.columns[6])
plt.title(label=rahang)
plt.xlabel('Epoch')
plt.legend()
plt.subplot(223)
plt.plot(df.index[-60:], df.iloc[-60:,3], label=df.columns[3])
plt.plot(df.index[-60:], df.iloc[-60:,7], label=df.columns[7])
plt.title(label=rahang)
plt.xlabel('Epoch')
plt.legend()
plt.subplot(224)
plt.plot(df.index[-60:], df.iloc[-60:,4], label=df.columns[4])
plt.plot(df.index[-60:], df.iloc[-60:,8], label=df.columns[8])
plt.title(label=rahang)
plt.xlabel('Epoch')
plt.legend()
plt.show()