import h5py
import numpy as np
from ECG_multi_lead_dataloader import *
import torchvision.transforms as tvtf
import transforms as tf
import os
import pickle
import matplotlib.pyplot as plt
from ECG_rendered_multilead_dataloader import *
from sys import getsizeof
from skimage import img_as_ubyte
import cv2
from ECG_renderer import *
import time
import random


##################################################################################################
# New database directory
target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
# ECG_test=ECG_Rendered_Multilead_Dataset(root_dir=target_path,transform=None,new_format=True) # For access demo

# with h5py.File(target_path+"Unified_rendered_db.hdf5","w") as f:
#     for x in range(len(ECG_test)):
#         to_store=ECG_test[x]
#         dset=f.create_dataset(str(x),data=to_store[0])
#         print(f"Processed : {x}")




# # Original directory
# root_dir = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data'+'\\'

# start = time.time()
# ECG_test=ECG_Multilead_Dataset(root_dir=root_dir,transform=None,new_format=False) # For access demo
# sample_test=ECG_test[2] #Taking for example record number 2 (Starting from zero)
# end = time.time()
# print(f'Time taken {end-start}')
# # start = time.time()
# # ECG_test=ECG_Multilead_Dataset(root_dir=target_path,transform=None,new_format=True) # For access demo
# # sample_test2=ECG_test[2] #Taking for example record number 2 (Starting from zero)
# # end = time.time()
# # print(f'Time taken {end-start}')
# print('Stopped here')

########################################################################################
# Converting the existing AF database to the new format
# Define the transforms that should be applied to each ECG record before returning it


# Reading the data


# f=h5py.File(target_path+'long_lead_data.hdf5', 'r')
# f_keys=f.keys()
# n1 = f.get('0')

# Writing the data
# file_divider=15
# ECG_test = ECG_Multilead_Dataset(root_dir=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data' + '\\',
#                                  transform=None) 
# for chunk_cntr in range(42):
#     print(f'Processing chunk # {chunk_cntr}')
#     short_leads_data=[]
#     long_lead_data=[]
#     diagnosis=[]
#     for cntr in range(min(1000,len(ECG_test)-chunk_cntr*1000)):
#         print(f'')
#         K=ECG_test[chunk_cntr*1000+cntr]
#         short_leads_data.append(K[0][0])
#         long_lead_data.append(K[0][1])
#         diagnosis.append(int(K[1]))


#     if chunk_cntr==0:
#         data=short_leads_data
#         with h5py.File(target_path+'short_leads_digitized'+str(chunk_cntr//file_divider)+'.hdf5', 'w') as f:
#             dset = f.create_dataset(str(chunk_cntr), data=data)
#         data=long_lead_data
#         with h5py.File(target_path+'long_lead_data_digitized'+str(chunk_cntr//file_divider)+'.hdf5', 'w') as f:
#             dset = f.create_dataset(str(chunk_cntr), data=data)
#         data=diagnosis
#         with h5py.File(target_path+'diagnosis_digitized'+str(chunk_cntr//file_divider)+'.hdf5', 'w') as f:
#             dset = f.create_dataset(str(chunk_cntr), data=data)
#     else:
#         data=short_leads_data
#         with h5py.File(target_path+'short_leads_digitized'+str(chunk_cntr//file_divider)+'.hdf5', 'a') as f:
#             dset = f.create_dataset(str(chunk_cntr), data=data)
#         data=long_lead_data
#         with h5py.File(target_path+'long_lead_data_digitized'+str(chunk_cntr//file_divider)+'.hdf5', 'a') as f:
#             dset = f.create_dataset(str(chunk_cntr), data=data)
#         data=diagnosis
#         with h5py.File(target_path+'diagnosis_digitized'+str(chunk_cntr//file_divider)+'.hdf5', 'a') as f:
#             dset = f.create_dataset(str(chunk_cntr), data=data)        

# print('Finished saving')
###############################################################################


# loaded_data=[]
# for cntr in range(2):
#     f=h5py.File(target_path+'rendered_db_'+str(cntr)+'.hdf5', 'r')
#     f_keys=f.keys()
#     for key in f_keys:
#         n1 = f.get(key)
#         loaded_data.append(np.array(n1))
# store_every=650
# storage=[]
# for record_cntr in range(41830):
#         if (record_cntr)%store_every==0 and record_cntr>0:
#                 print(f'Stored. record counter: {record_cntr}')
#                 storage.append(record_cntr)

# print(np.diff(storage))
# loaded_data=[]
# for cntr in range(3):
#     f=h5py.File(target_path+'long_lead_data_digitized'+str(cntr)+'.hdf5', 'r')
#     f_keys=f.keys()
#     for key in f_keys:
#         n1 = f.get(key)
#         loaded_data.append(np.array(n1))

# loaded_data1=[]
# for cntr in range(3):
#     f=h5py.File(target_path+'short_leads_digitized'+str(cntr)+'.hdf5', 'r')
#     f_keys=f.keys()
#     for key in f_keys:
#         n1 = f.get(key)
#         loaded_data1.append(np.array(n1))


# data_for_storage=[]
# store_every=650
# file_number=0
# filename_str='rendered_db_'
# images_stored=0
# for record_cntr in range(41830):
#         ECG_test=(np.array(loaded_data1[record_cntr//1000][record_cntr%1000]),np.array(loaded_data[record_cntr//1000][record_cntr%1000]))
#         Current_image = draw_ECG_multilead_vanilla(ECG_test)
#         Output=Current_image[15:690, 135:1585, :]
#         Output=img_as_ubyte(Output)
#         data_for_storage.append(Output)
#         images_stored+=1
#         print(f'Processing record number {record_cntr}')
#         if (record_cntr+1)%store_every==0 and record_cntr>0:
#                 filename=filename_str+str(record_cntr//store_every)+'.hdf5'
#                 print('Storing file: '+filename)
#                 with h5py.File(target_path+filename, 'w') as f:
#                         dset = f.create_dataset(str(record_cntr//store_every), data=data_for_storage)
#                 data_for_storage=[]
#                 print(f'Images stored {images_stored}')
#                 images_stored=0

#         if record_cntr==41829:
#                 filename=filename_str+str(record_cntr//store_every+1)+'.hdf5'
#                 print('Storing file: '+filename)
#                 with h5py.File(target_path+filename, 'w') as f:
#                         dset = f.create_dataset(str(record_cntr//store_every+1), data=data_for_storage)                
#                 print(f'Images stored {images_stored}')


# print('Reached the end')








# ##############################################################################
# print('Try to draw and save in real time')
# loaded_data_long_lead_data_digitized=[]
# f=h5py.File(target_path+'long_lead_data_digitized'+str(0)+'.hdf5', 'r')
# f_keys=f.keys()
# for key in f_keys:
#     n1 = f.get(key)
#     loaded_data_long_lead_data_digitized.append(np.array(n1))
# ECG_info=loaded_data_long_lead_data_digitized[0]
# X=draw_ECG_multilead_vanilla(ECG_info)



# root_dir = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Rendering_to_class_db' + r'\\'
# ECG_test = ECG_Rendered_Multilead_Dataset(root_dir=root_dir, transform=None, partial_upload=False)  # For KNN demo
# data=[]
# split_every=650
# for cntr in range(41830):
#     K = ECG_test[cntr]
#     img = img_as_ubyte(K[0])
#     data.append(img)
#     print(f'Processing : {cntr}')
#     if cntr>0 and cntr%split_every==0:
#         with h5py.File(target_path+'rendered_image'+str(cntr//split_every-1)+'.hdf5', 'w') as f:
#             dset = f.create_dataset(str(cntr//split_every), data=data)
#             data=[]



# ###############################################################################
# # Testing the written data
# loaded_data=[]
# for cntr in range(3):
#     f=h5py.File(target_path+'diagnosis_digitized'+str(cntr)+'.hdf5', 'r')
#     f_keys=f.keys()
#     for key in f_keys:
#         n1 = f.get(key)
#         loaded_data.append(np.array(n1))
 
# total_num=0
# for item in loaded_data:
#     total_num+=len(item)
# print(f'Finished loading, total number of diagnoses {total_num}')

# loaded_data=[]
# for cntr in range(3):
#     f=h5py.File(target_path+'long_lead_data_digitized'+str(cntr)+'.hdf5', 'r')
#     f_keys=f.keys()
#     for key in f_keys:
#         n1 = f.get(key)
#         loaded_data.append(np.array(n1))
 
# total_num=0
# for item in loaded_data:
#     total_num+=len(item)
# print(f'Finished loading, total number of long_lead_data_digitized {total_num}')



# loaded_data=[]
# for cntr in range(3):
#     f=h5py.File(target_path+'short_leads_digitized'+str(cntr)+'.hdf5', 'r')
#     f_keys=f.keys()
#     for key in f_keys:
#         n1 = f.get(key)
#         loaded_data.append(np.array(n1))
 
# total_num=0
# for item in loaded_data:
#     total_num+=len(item)
# print(f'Finished loading, total number of short_leads_digitized {total_num}')


# # #####  Investigate performance of the loader #####
# # Randomly choose 100 numbers
# ECG_test=ECG_Rendered_Multilead_Dataset(root_dir=target_path,transform=None,new_format=True) # For access demo
# sample_test=ECG_test[2] #Taking for example record number 2 (Starting from zero)
# num_of_records_to_test=50
# records_for_test=[]
# start = time.time()
# for x in range(num_of_records_to_test):
#     records_for_test.append(random.randint(0,len(ECG_test)))
#     sample_test=ECG_test[records_for_test[-1]]
#     print(f'Now processidng: {x}')

# end = time.time()
# print(f'Time taken {end-start}')

# ## New format
# start = time.time()
# for x in records_for_test:
#     print(f'Now processidng: {x}')
#     with h5py.File(target_path+  "Unified_rendered_db.hdf5", "r") as f:
#         n1=f.get(str(x))
#         image_data=np.array(n1)
# end = time.time()
# print(f'Time taken {end-start}')
        


######################  Split big database to 20 pieces ###################

target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
ECG_test=ECG_Rendered_Multilead_Dataset(root_dir=target_path,transform=None,new_format=True) # For access demo
ECG_test[41830]
num_of_records= len(ECG_test)
split_to_parts=20
every_part_is=num_of_records//split_to_parts
for part_cntr in range(split_to_parts+1):
        with h5py.File(target_path+"Unified_rendered_db_split"+str(part_cntr) + ".hdf5","w") as f:
                min_val=part_cntr*every_part_is
                max_val=min((part_cntr+1)*every_part_is,num_of_records)
                for x in range(min_val,max_val):
                        to_store=ECG_test[x]
                        dset=f.create_dataset(str(x),data=to_store[0])
                        print(f"Processed : {x}")

print('Finished')