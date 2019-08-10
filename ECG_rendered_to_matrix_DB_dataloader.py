from torch.utils.data import Dataset
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import random


class ECG_Rendered_to_matrix_Dataset(Dataset):
    # Convention   [n , height, width, color channel] 
    def __init__(self, root_dir=None, transform=None, partial_upload=False,new_format=True):
        super().__init__()
        self.data = []
        self.digitized_data=[]
        self.data_info = []
        self.transform = transform
        self.last_chunk_uploaded_to_memory = 1
        self.partial_upload = partial_upload
        self.new_format=new_format
        self.batch_size_in_file_new_format=650
        self.root_dir=root_dir        

        if root_dir is None:
            self.dataset_path = os.getcwd()+'\\Chineese_database\\'
        else:
            self.dataset_path = root_dir
        if new_format==False:
            for indx, file in enumerate(glob.glob(self.dataset_path+"*.pkl")):
                unpickled_data = self.unpickle_ECG_data(file=file)
                list_shape = np.shape(unpickled_data)
                self.samples = unpickled_data
                if indx == 0:  # (partial_upload) and
                    break
        else:
            short_leads_data=[]
            long_lead_data=[]
            image_data=[]
            self.last_chunk_uploaded_to_memory=0
            for cntr in range(3):
                f=h5py.File(root_dir+'short_leads_digitized'+str(cntr)+'.hdf5', 'r')
                f_keys=f.keys()
                for key in f_keys:
                    n1 = f.get(key)
                    short_leads_data.append(np.array(n1))

                f=h5py.File(root_dir+'long_lead_data_digitized'+str(cntr)+'.hdf5', 'r')
                f_keys=f.keys()
                for key in f_keys:
                    n1 = f.get(key)
                    long_lead_data.append(np.array(n1))
            for batch_cntr in range(len(long_lead_data)):
                for record_in_batch_cntr in range(len(long_lead_data[batch_cntr])):
                    self.digitized_data.append((short_leads_data[batch_cntr][record_in_batch_cntr],long_lead_data[batch_cntr][record_in_batch_cntr]))

            # f=h5py.File(root_dir+'rendered_db_'+str(0)+'.hdf5', 'r')
            # f_keys=f.keys()
            # for key in f_keys:
            #     n1 = f.get(key)
            #     image_data.append(np.array(n1))

            # self.batch_size_in_file_new_format=len(image_data[0])

            # for batch_cntr in range(len(image_data)):
            #     for record_in_batch_cntr in range(len(image_data[batch_cntr])):
            #         self.data.append((np.array(image_data[batch_cntr][record_in_batch_cntr]),self.digitized_data[batch_cntr*self.batch_size_in_file_new_format+record_in_batch_cntr]))
            
            # self.samples=self.data

        # print(f'Uploaded data, size of {np.shape(self.samples)}')

    def __len__(self):
        if self.new_format:
            return len(self.digitized_data)
        else:
            if self.partial_upload:
                return len(self.samples)
            else:
                return 41829

    def __getitem__(self, idx):
        #TODO: Implement transformation
        if self.new_format:
            # required_chunk=idx//self.batch_size_in_file_new_format
            # if not (required_chunk==self.last_chunk_uploaded_to_memory):
            #     image_data=[]                
            #     f=h5py.File(self.root_dir+'rendered_db_'+str(required_chunk)+'.hdf5', 'r')
            #     f_keys=f.keys()
            #     for key in f_keys:
            #         n1 = f.get(key)
            #         image_data.append(np.array(n1))  
            #     self.data=[]   
            #     for batch_cntr in range(len(image_data)):
            #         for record_in_batch_cntr in range(len(image_data[batch_cntr])):
            #             self.data.append((np.array(image_data[batch_cntr][record_in_batch_cntr]),self.digitized_data[idx]))
           
            # sample=(self.data[idx%self.batch_size_in_file_new_format])
            with h5py.File(self.root_dir+  "Unified_rendered_db.hdf5", "r") as f:
                n1=f.get(str(idx))
                image_data=np.array(n1)
            sample=(image_data,self.digitized_data[idx])
            return sample

        else:
            chunk_number = idx//1000+1
            if chunk_number == self.last_chunk_uploaded_to_memory:
                if self.transform:
                    sample = self.transform(self.samples[idx % 1000])
                else:
                    sample = self.samples[idx % 1000]
                return sample
            else:
                file = self.dataset_path+'Rendered_data'+str(max(1, idx//1000+1))+'.pkl'
                unpickled_data = self.unpickle_ECG_data(file=file)
                self.last_chunk_uploaded_to_memory = max(1, idx//1000+1)
                self.samples = unpickled_data
                if self.transform:
                    sample = self.transform(self.samples[idx % 1000])
                else:
                    sample = self.samples[idx % 1000]
                return sample

    @staticmethod
    def unpickle_ECG_data(file='ECG_data.pkl'):
        with open(file, 'rb') as fo:
            pickled_data = pickle.load(fo, encoding='bytes')
        print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data  

    def plot(self, idx):
        item_to_show = self.__getitem__(idx)
        ax1 = plt.subplot(211)
        plt.imshow(item_to_show[0])
        ax2 = plt.subplot(223)
        plt.imshow(item_to_show[1][0])
        ax3 = plt.subplot(224)
        plt.plot(np.squeeze(item_to_show[1][1]))
        ax1.title.set_text('Image')
        ax2.title.set_text('Matrix')
        ax3.title.set_text('Long lead')
        plt.show()
        return


if __name__ == "__main__":
    # New database directory
    target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
    ECG_test = ECG_Rendered_to_matrix_Dataset(root_dir=target_path, transform=None, partial_upload=False)  # For KNN demo
    testing_array=[2000]#list(range(2040,2050))
    start = time.time()
    for x in range(1000):
        r=random.randint(0,len(ECG_test))
        K=ECG_test[r]

    end=time.time()
    print(f'It took {end-start} sec.')

    for indx in testing_array:
        K = ECG_test[indx]
        plt.imshow(K[0])
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
        plt.plot(K[1][1][0])
        plt.show()
        # print(f'Is AFIB: {K[1]}')