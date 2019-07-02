#%% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob,os
import scipy.io as sio
import numpy as np
import pandas as pd

#%% Chineese challenge
def Upload_db_records(DB_path):
    for file in glob.glob("*.mat"):
        print(file)
        mat_contents = sio.loadmat(DB_path+file)
        B=mat_contents['ECG']['data'].item() 
        fig, axes = plt.subplots(nrows=6, ncols=2)
        DB_ref_path=DB_path+'REFERENCE.csv'

        classification=upload_classification(DB_ref_path,file)     
        fig.suptitle(f'Record number {file}, Is AFIB: {classification}')
        for ax, cntr in zip(axes.flatten(),range(12)):
            ax.plot(B[cntr,:],linewidth=1.0)
            ax.set(title=titles[cntr])

        plt.plot()
        plt.show()

#%% Upload DB classification
def upload_classification(DB_ref_path,required_entry):
    print(DB_ref_path)
    data = pd.read_csv(DB_ref_path)
    data.head()
    _entries=data.Recording.to_list()
    _entry_number_in_list=_entries.index(required_entry[0:5])
    _values=data.values[_entry_number_in_list,:]
    if (2 in _values):
        classification=True
    else:
        classification=False
    return classification
    
#%% Main loop
DB_path=r'C:\Source_Control_Map_Git\2366605-Final-Project\Data\Original\Chineese\\'
titles=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
os.chdir(DB_path)
Upload_db_records(DB_path)