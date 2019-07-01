#%% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob,os
import scipy.io as sio
import numpy as np

#%% Chineese challenge
DB_path=r'C:\Source_Control_Map_Git\2366605-Final-Project\Data\Original\Chineese\\'
os.chdir(DB_path)
for file in glob.glob("*.mat"):
    print(file)
    mat_contents = sio.loadmat(DB_path+file)
    B=mat_contents['ECG']['data'].item() 
    p=620
    for cntr in range(9):
        plt.subplot(p+cntr+1)
        plt.plot(B[cntr,:])
        plt.ylabel(f'Lead {cntr+1}')
    plt.plot()
    plt.show()
