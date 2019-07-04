#%% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob,os
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import pickle


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
        Relevant_Data=ECG_raw[:,int(2.5*(record_cntr)*Fs):int(2.5*(record_cntr+1)*Fs)]
        Scaled_data=(Relevant_Data-Relevant_Data.min())/(Relevant_Data.max()-Relevant_Data.min())
        print('Here')
    # Return tuple of (2.5 sec X 12 lead matrix + one strip of 10 records)
    return 

def unpickle_CIFAR_dataset(file):
    """ Upolading CIFAR hust to see 2the convention
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    
    """    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def pickle_ECG_data(ECG_data):
    with open('ECG_data.pkl','wb') as fo:
        pickle.dump(ECG_data,fo,-1) # Pickling with th highest protocol available
    
def unpickle_ECG_data(file='ECG_data.pkl'):
    with open(file, 'rb') as fo:
        pickled_data = pickle.load(fo, encoding='bytes')
    print(f'Loaded data with type of: {type(pickled_data)}')
    return pickled_data    

#%% Main loop
returned_dict=unpickle_CIFAR_dataset(r'C:\Users\vgliner\AppData\Local\Temp\1\cifar-10-python\cifar-10-batches-py\data_batch_1')
# Pickle test
pickle_ECG_data('Vadim')
unpickle_ECG_data()
cwd = os.getcwd()
DB_path=cwd+r'\Data\Original\Chineese'+'\\'
titles=['Lead1','Lead2','Lead3','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
os.chdir(DB_path)
Upload_db_records(DB_path,plot_flag=False)
