import numpy as np
import cv2
from ECG_renderer import *
from ECG_multi_lead_dataloader import *
import matplotlib.pyplot as plt
from ECG_pickling import *

num_of_chunks= 42
list_of_records_statistics=[]
total_records_num=0
for cntr in range(num_of_chunks):
    print(f'Checking chunk # {cntr}')
    file='C:\\Users\\vgliner\\OneDrive - JNJ\\Desktop\\Data\\Rendering_to_class_db\\'+"Rendered_data"+str(cntr+1)+".pkl"
    unpickled_data=unpickle_ECG_data(file=file)
    list_shape=np.shape(unpickled_data)
    list_of_records_statistics.append([len(unpickled_data), list_shape[0],list_shape[1] ])
    total_records_num+=len(unpickled_data)
    print(f'Shape: {list_of_records_statistics[-1]}, total records number so far: {total_records_num}')


ECG_test=ECG_Multilead_Dataset(root_dir=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data'+'\\',transform= None, partial_upload=False) 
print(f'Original data length is {len(ECG_test)}')
print('Finished')