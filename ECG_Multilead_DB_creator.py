#%% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob,os
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import pickle
from ECG_pickling import *


#%% Chineese challenge
def Upload_db_records(DB_path, plot_flag=True):
    db_splitted_records=[]
    records_per_file=4000
    for file in glob.glob("*.mat"):
        print(file)
        mat_contents = sio.loadmat(DB_path+file)
        B=mat_contents['ECG']['data'].item() 
        DB_ref_path=DB_path+'REFERENCE.csv'

        classification=upload_classification(DB_ref_path,file)  
        ## Plotting, if necessary
        if plot_flag:   
            fig, axes = plt.subplots(nrows=6, ncols=2)
            fig.suptitle(f'Record number {file}, Is AFIB: {classification}')
            for ax, cntr in zip(axes.flatten(),range(12)):
                ax.plot(B[cntr,:],linewidth=1.0)
                ax.set(title=titles[cntr])
            plt.plot()
            plt.show()
        splitted_records=split_records(B)
        if splitted_records==-1:
            continue
        ## Splitting to a standard records
        for splitted_record in splitted_records:
            db_splitted_records.append((splitted_record,classification))
    for file_num in range(len(db_splitted_records)//records_per_file+1):
        max_storage_value=min([len(db_splitted_records),(file_num+1)*records_per_file])
        filename=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Chinese_db'+str(file_num)+'.pkl'
        pickle_ECG_data(db_splitted_records[file_num*records_per_file:max_storage_value],file=filename)
        print(f'Pickled: file Chinese_db_{file_num}.pkl')
    print(f'Created {len(db_splitted_records)} records')
    return db_splitted_records

#%% Upload DB classification
def upload_classification(DB_ref_path,required_entry):
    #print(DB_ref_path)
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
    ## Definitions
    Fs=500 # Hz
    bit_encoding=8 #bit
    Leads=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    Long_lead_type='Lead2'
    Long_lead_indx=Leads.index(Long_lead_type)
    Scale_type=1 # 0- No scaling, 1- Normalized between 0-1, 2- Normalized between 0- bit_encoding
    results_list=[]
    ## Calculations
    num_of_seconds_in_record=len(ECG_raw[0,:])/Fs
    if (num_of_seconds_in_record<10):
        return -1
    number_of_output_records=int(np.floor(num_of_seconds_in_record/2.5))
    for record_cntr in range(number_of_output_records):
        # Scale and quantize the records
        Relevant_Data=ECG_raw[:,int(2.5*(record_cntr)*Fs):int(2.5*(record_cntr+1)*Fs)]
        if (record_cntr<number_of_output_records-4):  # Taking according to the first 2.5 sec. 
            Long_lead_recording=ECG_raw[Long_lead_indx,int(2.5*(record_cntr)*Fs):int(2.5*(record_cntr+4)*Fs)]
        else:
            Long_lead_recording=ECG_raw[Long_lead_indx,int(2.5*(number_of_output_records-4)*Fs):int(2.5*(number_of_output_records)*Fs)]

        Scaled_data=Relevant_Data
        Scaled_data_long_lead=Long_lead_recording

        if (Long_lead_recording.max()-Long_lead_recording.min())<=0:
            return -1

        if (Relevant_Data.max()-Relevant_Data.min())<=0:
            return -1            

        if (Scale_type>0) :
            Scaled_data=(Relevant_Data-Relevant_Data.min())/(Relevant_Data.max()-Relevant_Data.min())
            Scaled_data_long_lead=(Long_lead_recording-Long_lead_recording.min())/(Long_lead_recording.max()-Long_lead_recording.min())

        if (Scale_type>1) :
            Scaled_data=(Scaled_data*(2**bit_encoding-1)).astype(int)
            Scaled_data_long_lead=(Scaled_data_long_lead*(2**bit_encoding-1)).astype(int)

        results_list.append((Scaled_data,np.expand_dims(Scaled_data_long_lead,axis=0)))
        #print(f'Record number {record_cntr} out of  {number_of_output_records} , long record length {len(Long_lead_recording)}, total : {len(ECG_raw[0,:])}')
    # Return tuple of (2.5 sec X 12 lead matrix + one strip of 10 records)
    return results_list



#%% Main loop
# returned_dict=unpickle_CIFAR_dataset(r'data_batch_1')
# Pickle test
# pickle_ECG_data('Vadim')
# unpickle_ECG_data()
cwd = os.getcwd()
DB_path=cwd+r'\Data\Original\Chineese'+'\\'
DB_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\SW\Chinese Challenge\Data - Original'+'\\'
titles=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
os.chdir(DB_path)
Upload_db_records(DB_path,plot_flag=False)
