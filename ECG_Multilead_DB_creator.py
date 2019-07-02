#%% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob,os
import scipy.io as sio
import numpy as np

#%% Chineese challenge
DB_path=r'C:\Source_Control_Map_Git\2366605-Final-Project\Data\Original\Chineese\\'
titles=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
os.chdir(DB_path)
for file in glob.glob("*.mat"):
    print(file)
    mat_contents = sio.loadmat(DB_path+file)
    B=mat_contents['ECG']['data'].item() 
    fig, axes = plt.subplots(nrows=6, ncols=2)
    fig.suptitle(f'Record number {file}')
    for ax, cntr in zip(axes.flatten(),range(12)):
        ax.plot(B[cntr,:])
        ax.set(title=titles[cntr])
    plt.plot()
    plt.show()

