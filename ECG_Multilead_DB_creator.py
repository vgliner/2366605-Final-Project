#%% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob,os
import scipy.io as sio
import numpy as np
import pandas as pd
import os

#%% Chineese challenge
def Upload_db_records(DB_path, plot_flag=True):
    for file in glob.glob("*.mat"):
        print(file)
        mat_contents = sio.loadmat(DB_path+file)
        B=mat_contents['ECG']['data'].item() 
        fig, axes = plt.subplots(nrows=6, ncols=2)
        DB_ref_path=DB_path+'REFERENCE.csv'

        classification=upload_classification(DB_ref_path,file)  
        ## Plotting, if necessary
        if plot_flag:   
            fig.suptitle(f'Record number {file}, Is AFIB: {classification}')
            for ax, cntr in zip(axes.flatten(),range(12)):
                ax.plot(B[cntr,:],linewidth=1.0)
                ax.set(title=titles[cntr])
            plt.plot()
            plt.show()
        split_records(B)
        ## Splitting to a stnadard records

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

def split_records(ECG_raw):
    Fs=500 # Hz
    num_of_seconds_in_record=len(ECG_raw[0,:])/Fs
    if (num_of_seconds_in_record<10):
        return -1
    number_of_output_records=np.ceil(num_of_seconds_in_record/2.5)
    for record_cntr in range(int(number_of_output_records)-1):
        # Scale and quantize the records
        Relevant_Data=ECG_raw[:,(record_cntr)*Fs:(record_cntr+1)*Fs]
        Scaled_data=(Relevant_Data-Relevant_Data.min())/(Relevant_Data.max()-Relevant_Data.min())
        print('Here')
    # Return tuple of (2.5 sec X 12 lead matrix + one strip of 10 records)
    return

#%% Main loop
cwd = os.getcwd()
DB_path=cwd+r'\Data\Original\Chineese'+'\\'
titles=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
os.chdir(DB_path)
Upload_db_records(DB_path,plot_flag=False)