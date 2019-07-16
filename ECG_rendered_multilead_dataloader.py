from torch.utils.data import Dataset
import glob,os
import pickle
import numpy as np
#TODO: Adjust file tobe a class of the actual dataset

class ECG_Rendered_Multilead_Dataset(Dataset):
    # Convention   [n , height, width, color channel] 
    def __init__(self, root_dir= None, transform = None, partial_upload=False):
        super().__init__()
        self.data=[]
        self.data_info=[]
        self.transform= transform


        if root_dir== None:
            self.dataset_path=os.getcwd()+'\\Chineese_database\\'
        else:
            self.dataset_path=root_dir

        for indx, file in enumerate(glob.glob(self.dataset_path+"*.pkl")):
            unpickled_data=self.unpickle_ECG_data(file=file)
            list_shape=np.shape(unpickled_data)
            if len(list_shape)>1:
                self.data=self.data+unpickled_data
            else:
                self.data_info.append(unpickled_data)
            
            if (partial_upload) and (indx==0):
                break
        self.samples = self.data
        data=[d[0] for d in self.data]
        self.data=data
        self.target=[d[1] for d in self.data]
        print(f'Uploaded data, size of {np.shape(self.data)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.samples[idx])
        return self.samples[idx]


    def unpickle_ECG_data(self,file='ECG_data.pkl'):
        with open(file, 'rb') as fo:
            pickled_data = pickle.load(fo, encoding='bytes')
        print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data  

